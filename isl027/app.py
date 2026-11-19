import os
import re
import threading
import webbrowser
from typing import List
import pandas as pd

import gradio as gr

# Set environment variables to ensure the app runs locally and uses cached models without network requests
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_utils import analyze_match, extract_text_from_pdf, load_model
from nlp_features import (
    check_ats_compatibility, extract_top_keywords, generate_cover_letter,
    generate_feedback, get_learning_recommendations, create_keyword_chart,
    generate_interview_questions, detect_fake_job, create_skill_gap_chart,
    create_comparison_chart
)

# Pre-load the model into memory when the app starts to avoid delays on the first request
print("Loading NLP models from local cache, please wait...")
load_model()
print("Models loaded successfully! Starting app...")

def get_file_path(file) -> str:
    """Returns a usable file path for both Gradio 3 (file object) and Gradio 4+ (string path)."""
    if file is None:
        return ""
    return file.name if hasattr(file, "name") else str(file)

def highlight_skills(text: str, skills: List[str]) -> str:
    """Wraps the matched skills inside the CV text with HTML span tags to highlight them."""
    if not text:
        return ""
    highlighted = text
    for skill in skills:
        # Use regex with word boundaries to ensure exact matches
        pattern = re.compile(r"(?<!\w)(" + re.escape(skill) + r")(?!\w)", re.IGNORECASE)
        highlighted = pattern.sub(
            r'<span style="background-color: #d4edda; color: #155724; font-weight: bold; padding: 2px 4px; border-radius: 4px;">\1</span>',
            highlighted
        )
    return highlighted

def run_analysis(cv_text: str, jd_text: str, cv_file) -> tuple:
    """Main function to analyze a single CV against a JD."""
    try:
        if cv_file is not None:
            pdf_text = extract_text_from_pdf(get_file_path(cv_file))
            cv_text = (cv_text or "") + "\n" + pdf_text

        if not cv_text or not jd_text:
            return "Please provide both a CV and a Job Description.", "", "", None, "", "", "", None, "", "", "", ""

        result = analyze_match(cv_text, jd_text)
        
        score_md = f"""
        <div style='text-align: center; padding: 20px; background: rgba(37, 99, 235, 0.1); border-radius: 10px; border: 1px solid #2563eb;'>
            <p style='margin: 0; font-size: 1.1rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;'>Match Score</p>
            <h1 style='margin: 10px 0; font-size: 4rem; color: #3b82f6; font-weight: 800;'>{result['score']}%</h1>
            <p style='margin: 0; font-size: 1.2rem; color: #e2e8f0; font-weight: bold;'>{result['level']}</p>
        </div>
        """
        
        matched_html = "".join([f"<span style='display:inline-block; background: #10b981; color: white; padding: 4px 10px; border-radius: 15px; margin: 4px; font-size: 0.9rem; font-weight: 600;'>{s.capitalize()}</span>" for s in result['matched']])
        missing_html = "".join([f"<span style='display:inline-block; background: #ef4444; color: white; padding: 4px 10px; border-radius: 15px; margin: 4px; font-size: 0.9rem; font-weight: 600;'>{s.capitalize()}</span>" for s in result['missing']])
        
        matched_md = f"### Matched Skills\n<div>{matched_html if matched_html else 'None found'}</div>"
        missing_md = f"### Missing Skills\n<div>{missing_html if missing_html else 'None detected'}</div>"
        
        gap_chart_fig = create_skill_gap_chart(result['matched'], result['missing'])
        scam_md = detect_fake_job(jd_text)
        ats_md = check_ats_compatibility(cv_text, result['missing'], result['matched'])
        keyword_md = extract_top_keywords(jd_text)
        keyword_chart_fig = create_keyword_chart(jd_text)
        questions_md = generate_interview_questions(result['matched'])
        
        feedback_md = f"### Automated Feedback\n{generate_feedback(result['score'], result['missing'])}"
        learning_md = f"### Skill Gap Learning Path\n{get_learning_recommendations(result['missing'])}"
        
        hl_text = highlight_skills(cv_text, result['matched'])
        highlight_md = (
            "### Document Preview\n"
            f"<div style='max-height: 300px; overflow-y: auto; padding: 15px; border: 1px solid #475569; "
            f"background-color: #1e293b; border-radius: 8px; white-space: pre-wrap; font-size: 14px; line-height: 1.6; color: #cbd5e1;'>{hl_text}</div>"
        )
        
        return score_md, matched_md, missing_md, gap_chart_fig, scam_md, ats_md, keyword_md, keyword_chart_fig, questions_md, feedback_md, learning_md, highlight_md
        
    except Exception as e:
        return f"Error: {str(e)}", "", "", None, "", "", "", None, "", "", "", ""

def run_batch_analysis(jd_text: str, files: list):
    """
    Handles multiple CVs for Recruiter Mode.
    It ranks all candidates and automatically highlights the top 3 (or fewer if fewer are provided).
    """
    if not jd_text or not files:
        return None, pd.DataFrame([["Error", "Please provide JD and at least one CV", "", "", ""]], columns=["Candidate", "Score", "Details", "Level", "Status"])
    
    results = []
    names = []
    scores = []
    
    # Step 1: Process each uploaded PDF file and calculate the match score
    for file in files:
        try:
            file_path = get_file_path(file)
            pdf_text = extract_text_from_pdf(file_path)
            result = analyze_match(pdf_text, jd_text)

            candidate_name = os.path.basename(file_path).replace(".pdf", "")
            matched_str = ", ".join(result['matched']) if result['matched'] else "None"
            
            results.append({
                "name": candidate_name,
                "score": result['score'],
                "matched": matched_str,
                "level": result['level']
            })
        except Exception as e:
            results.append({
                "name": os.path.basename(get_file_path(file)),
                "score": 0.0,
                "matched": "Error reading file",
                "level": "Error"
            })
            
    # Step 2: Sort all candidates by their match score in descending order (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Step 3: Prepare the table data. Dynamically label the top 3 candidates.
    final_table_data = []
    for index, row in enumerate(results):
        # Automatically assign a rank based on their position after sorting
        if index == 0:
            status = "Top 1 (Best Match)"
        elif index == 1:
            status = "Top 2"
        elif index == 2:
            status = "Top 3"
        else:
            status = "Standard Match"
            
        final_table_data.append([row['name'], row['score'], row['matched'], row['level'], status])
        
        # Collect data for the bar chart
        names.append(row['name'])
        scores.append(row['score'])
    
    # Step 4: Create a DataFrame containing ALL candidates, sorted by score
    df = pd.DataFrame(
        final_table_data, 
        columns=["Candidate Name", "Match Score (%)", "Matched Skills", "Level", "Recruiter Status"]
    )
    
    # Step 5: Create a chart showing all candidates, with the top 3 highlighted in green
    chart_fig = create_comparison_chart(names, scores, top_n=3)
    
    return chart_fig, df

def run_cover_letter(cv_text: str, jd_text: str, skills_text: str) -> str:
    """Generate a cover letter using the supplied CV, JD, and supported skills."""
    if not cv_text or not cv_text.strip():
        return "Please provide your resume text."

    if not jd_text or not jd_text.strip():
        return "Please provide the job description."

    try:
        match_result = analyze_match(cv_text, jd_text)
        selected_skills = list(match_result["matched"])

        manual_skills = [
            skill.strip()
            for skill in (skills_text or "").split(",")
            if skill.strip()
        ]

        # Only keep manually requested skills that are actually present in the resume.
        supported_manual_skills = [
            skill
            for skill in manual_skills
            if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", cv_text, re.IGNORECASE)
        ]

        selected_skills = list(dict.fromkeys(selected_skills + supported_manual_skills))

        return generate_cover_letter(cv_text, jd_text, selected_skills)

    except Exception as error:
        return f"Could not generate the cover letter: {error}"

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

# Build the main UI
with gr.Blocks(theme=theme, title="Resume Optimization System") as demo:
    
    gr.Markdown("# Resume Optimization System\n**Advanced NLP Analysis for Job Seekers and Professionals**")
    
    with gr.Tabs():
        # --- TAB 1: Resume Analyzer ---
        with gr.TabItem("Resume Analyzer"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 1. Input Data")
                        cv_text = gr.Textbox(label="Resume Text", lines=6, placeholder="Paste your full CV text here...")
                        cv_file = gr.File(label="Or upload Resume as PDF", file_types=[".pdf"])
                        jd_text = gr.Textbox(label="Job Description", lines=6, placeholder="Paste the target job description here...")
                        
                    with gr.Row():
                        btn = gr.Button("Analyze Match", variant="primary", size="lg")
                        stop1 = gr.Button("Clear", variant="secondary", size="lg")

                with gr.Column(scale=1):
                    with gr.Group():
                        out_score = gr.HTML() 
                        with gr.Row():
                            out_matched = gr.Markdown()
                            out_missing = gr.Markdown()
                        out_gap_chart = gr.Plot(show_label=False)
                    
                    with gr.Group():
                        out_scam = gr.Markdown()
                    
                    with gr.Group():
                        gr.Markdown("### Keyword Frequency Chart")
                        out_keyword_chart = gr.Plot(show_label=False)
                    
                    with gr.Accordion("ATS Compatibility & Keyword List", open=False):
                        with gr.Group():
                            out_ats = gr.Markdown()
                            out_keywords = gr.Markdown()
                    
                    with gr.Accordion("Interview Preparation", open=False):
                        out_questions = gr.Markdown()

                    with gr.Accordion("Feedback & Learning Recommendations", open=False):
                        with gr.Group():
                            out_feedback = gr.Markdown()
                            out_learning = gr.Markdown()

                    with gr.Accordion("Document Preview (Highlighted Skills)", open=False):
                        out_highlight = gr.Markdown()

            event1 = btn.click(
                fn=run_analysis, 
                inputs=[cv_text, jd_text, cv_file], 
                outputs=[out_score, out_matched, out_missing, out_gap_chart, out_scam, out_ats, out_keywords, out_keyword_chart, out_questions, out_feedback, out_learning, out_highlight]
            )
            stop1.click(fn=lambda: ("", "", "", None, "", "", "", None, "", "", "", ""), inputs=None, outputs=[out_score, out_matched, out_missing, out_gap_chart, out_scam, out_ats, out_keywords, out_keyword_chart, out_questions, out_feedback, out_learning, out_highlight])

        # --- TAB 2: Cover Letter Generator ---
        with gr.TabItem("Cover Letter Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Letter Details")
                        cl_cv = gr.Textbox(label="Resume Text", lines=5)
                        cl_jd = gr.Textbox(label="Job Description", lines=5)
                        cl_skills = gr.Textbox(label="Key Skills to Highlight (comma separated)", lines=2, placeholder="e.g., Python, SQL, Machine Learning")
                        
                        with gr.Row():
                            cl_btn = gr.Button("Generate Letter", variant="primary", size="lg")
                            stop2 = gr.Button("Clear", variant="secondary", size="lg")
                
                with gr.Column(scale=1):
                    with gr.Group():
                        cl_output = gr.Textbox(label="Generated Cover Letter Draft", lines=16)
            
            event2 = cl_btn.click(
                fn=run_cover_letter, 
                inputs=[cl_cv, cl_jd, cl_skills], 
                outputs=[cl_output]
            )
            stop2.click(fn=lambda: ("",), inputs=None, outputs=[cl_output])

        # --- TAB 3: Recruiter Mode (Batch Compare) ---
        with gr.TabItem("Recruiter Mode"):
            gr.Markdown("### Batch Candidate Analysis\nUpload multiple PDF resumes against a single Job Description. The system will automatically rank all candidates and highlight the top matches.")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_jd = gr.Textbox(label="Job Description", lines=6, placeholder="Paste the JD here...")
                    batch_files = gr.File(label="Upload Multiple Resumes (PDF)", file_count="multiple", file_types=[".pdf"])
                    
                    
                    with gr.Row():
                        batch_btn = gr.Button("Compare Candidates", variant="primary", size="lg")
                        stop3 = gr.Button("Clear", variant="secondary", size="lg")
                
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Comparison Chart (Top Candidates Highlighted in Green)")
                        batch_chart = gr.Plot(show_label=False)
                    
                    with gr.Group():
                        gr.Markdown("### Candidate Ranking Table (Sorted from Best to Lowest)")
                        batch_table = gr.Dataframe(
                            headers=["Candidate Name", "Match Score (%)", "Matched Skills", "Level", "Recruiter Status"],
                            datatype=["str", "number", "str", "str", "str"],
                            label=""
                        )

            # Updated to only take JD and Files as inputs (no slider)
            batch_btn.click(fn=run_batch_analysis, inputs=[batch_jd, batch_files], outputs=[batch_chart, batch_table])
            stop3.click(fn=lambda: (None, pd.DataFrame()), inputs=None, outputs=[batch_chart, batch_table])

if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:7860")
    
    threading.Timer(2, open_browser).start()
    demo.launch(server_name="127.0.0.1", server_port=7860)