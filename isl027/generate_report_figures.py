#!/usr/bin/env python3
"""
Professional Publication-Quality Report Figures Generator
for Resume Optimization System.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)

# Elegant professional color palette (Corporate / Academic Tech Report)
NAVY = "#0F172A"         # Dark slate / almost black for primary text
SLATE = "#334155"        # Neutral slate for secondary text
MUTED = "#64748B"        # Light slate for captions / metadata
BORDER = "#CBD5E1"       # Subtle card border

PRIMARY = "#1E40AF"      # Deep Royal Blue
PRIMARY_LIGHT = "#EFF6FF"# Soft Blue background
PRIMARY_BORDER = "#93C5FD"

SECONDARY = "#0D9488"    # Teal
SECONDARY_LIGHT = "#F0FDFA"
SECONDARY_BORDER = "#99F6E4"

ACCENT = "#D97706"       # Amber / Gold
ACCENT_LIGHT = "#FFFBEB"
ACCENT_BORDER = "#FDE68A"

PURPLE = "#7C3AED"       # Violet / Purple
PURPLE_LIGHT = "#F5F3FF"
PURPLE_BORDER = "#DDD6FE"

CARD_BG = "#FFFFFF"
CANVAS_BG = "#F8FAFC"    # Very light cool grey

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.facecolor": CANVAS_BG,
    "savefig.facecolor": CANVAS_BG,
})


def create_canvas(title: str, subtitle: str, figsize=(13, 8.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header section
    ax.text(3.5, 96.5, title, fontsize=18, weight="bold", color=NAVY, va="top")
    ax.text(3.5, 92.5, subtitle, fontsize=9.5, color=MUTED, va="top")
    
    # Decorative header rule
    ax.plot([3.5, 96.5], [90.5, 90.5], color="#E2E8F0", lw=1.2)
    return fig, ax


def draw_card(ax, x, y, w, h, title, subtitle="",
              bg=CARD_BG, border=BORDER, title_color=NAVY,
              sub_color=SLATE, title_size=8.5, sub_size=7.2,
              rounding=1.2, lw=1.2):
    """Draws a crisp modern rounded card."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.2,rounding_size={rounding}",
        linewidth=lw, edgecolor=border, facecolor=bg
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center",
                fontsize=sub_size, color=sub_color)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=title_color)


def draw_badge(ax, x, y, w, h, text, bg=PRIMARY, text_color="#FFFFFF", size=7.5):
    """Draws a pill/badge label for categories."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        linewidth=0, facecolor=bg
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, weight="bold", color=text_color)


def draw_arrow(ax, start, end, color=MUTED, lw=1.3, rad=0.0, head_size=11, dashed=False):
    style = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=head_size,
        linewidth=lw, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2
    )
    ax.add_patch(style)


def save_fig(fig, filename):
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(OUT / filename.replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Generated: {OUT / filename}")


# ==============================================================================
# FIGURE 1: SYSTEM ARCHITECTURE
# ==============================================================================
def architecture():
    fig, ax = create_canvas(
        "System Architecture",
        "Layered modular architecture of the Resume Optimization System",
        figsize=(13, 8.5)
    )

    layers = [
        {
            "badge": "PRESENTATION LAYER (GRADIO UI)",
            "color": PRIMARY,
            "bg": PRIMARY_LIGHT,
            "border": PRIMARY_BORDER,
            "y": 74.5, "h": 12.5,
            "cards": [
                ("Resume Analyzer", "CV & JD matching, skill gap analysis, ATS score"),
                ("Cover Letter Generator", "AI draft customized by matched & target skills"),
                ("Recruiter Mode", "Multi-resume batch PDF ranking & candidate comparison")
            ]
        },
        {
            "badge": "APPLICATION LOGIC & CONTROLLER",
            "color": SECONDARY,
            "bg": SECONDARY_LIGHT,
            "border": SECONDARY_BORDER,
            "y": 57.5, "h": 12.5,
            "cards": [
                ("Input Preprocessing", "PDF extraction (pypdf), text cleaning & normalization"),
                ("Workflow Coordinator", "Gradio Tab routing, data validation & error handling"),
                ("Analytics Engine", "Skill frequency counting, ATS scoring, scam detection")
            ]
        },
        {
            "badge": "NLP & MACHINE LEARNING CORE",
            "color": ACCENT,
            "bg": ACCENT_LIGHT,
            "border": ACCENT_BORDER,
            "y": 40.5, "h": 12.5,
            "cards": [
                ("Rule-based Skill Extractor", "Regex boundary matching over 48+ tech skills dictionary"),
                ("Multilingual SentenceTransformer", "paraphrase-multilingual-MiniLM-L12-v2 (dense embeddings)"),
                ("Cosine Semantic Similarity", "Vector dot product measuring deep contextual similarity")
            ]
        },
        {
            "badge": "SCORING & OUTPUT GENERATION",
            "color": PURPLE,
            "bg": PURPLE_LIGHT,
            "border": PURPLE_BORDER,
            "y": 23.5, "h": 12.5,
            "cards": [
                ("Blended Match Scorer", "0.60 × Semantic Sim + 0.40 × Skill Ratio (clamped 0-100%)"),
                ("Data Visualization Engine", "Skill gap pie chart, keyword frequency & candidate bar chart"),
                ("Mock Interview & Coaching", "Dynamic Easy/Medium/Hard questions & resource recommendations")
            ]
        },
        {
            "badge": "EVALUATION & VALIDATION BENCHMARK",
            "color": SLATE,
            "bg": "#F1F5F9",
            "border": BORDER,
            "y": 6.5, "h": 12.5,
            "cards": [
                ("Skill Extraction Benchmark", "Precision, Recall & F1-Score tested against gold labels"),
                ("Candidate Match Testing", "End-to-end verification (Candidate A, B, C profiles)"),
                ("Heuristic Scam Evaluation", "Confusion matrix & accuracy verification for fake job alerts")
            ]
        }
    ]

    for layer in layers:
        y = layer["y"]
        h = layer["h"]

        # Outer container for the tier
        outer = FancyBboxPatch(
            (3.5, y), 93, h,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            linewidth=1.2, edgecolor=layer["border"], facecolor=layer["bg"]
        )
        ax.add_patch(outer)

        # Layer badge
        draw_badge(ax, 5.5, y + h - 3.2, 34, 2.5,
                   layer["badge"], bg=layer["color"], text_color="#FFFFFF", size=7.2)

        # Cards inside tier
        cards = layer["cards"]
        n = len(cards)
        gap = 1.8
        total_w = 89
        w = (total_w - gap * (n - 1)) / n
        start_x = 5.5

        for i, (title, sub) in enumerate(cards):
            cx = start_x + i * (w + gap)
            draw_card(
                ax, cx, y + 1.2, w, h - 5.2,
                title=title, subtitle=sub,
                bg="#FFFFFF", border=layer["border"],
                title_color=NAVY, sub_color=SLATE,
                title_size=8.0, sub_size=6.6,
                rounding=0.8, lw=1.0
            )

    # Inter-tier connecting indicators (clean center flow)
    for y_top in [74.5, 57.5, 40.5, 23.5]:
        draw_arrow(ax, (50, y_top), (50, y_top - 4.5), color=SLATE, lw=1.4, head_size=10)

    save_fig(fig, "figure_01_system_architecture.png")


# ==============================================================================
# FIGURE 2: RESUME ANALYSIS PROCESS (CLEAN DUAL-STREAM)
# ==============================================================================
def analysis_process():
    fig, ax = create_canvas(
        "Resume Analysis & Scoring Process",
        "Dual-stream pipeline combining rule-based skill extraction with deep semantic embeddings",
        figsize=(13, 8.5)
    )

    # 1. Inputs (Top Row)
    draw_card(ax, 10, 77, 36, 10, "CANDIDATE RESUME (CV)",
              "Pasted text or extracted text-based PDF",
              bg=PRIMARY_LIGHT, border=PRIMARY_BORDER, title_color=PRIMARY)
    draw_card(ax, 54, 77, 36, 10, "JOB DESCRIPTION (JD)",
              "Target job requirements & qualification text",
              bg=SECONDARY_LIGHT, border=SECONDARY_BORDER, title_color=SECONDARY)

    # Preprocessing
    draw_card(ax, 10, 63, 36, 8.5, "CV Preprocessing",
              "PDF text extraction (pypdf), lowercase normalization & cleaning",
              bg="#FFFFFF", border=BORDER, title_size=8.2, sub_size=6.7)
    draw_card(ax, 54, 63, 36, 8.5, "JD Preprocessing",
              "Lowercase normalization & whitespace cleaning",
              bg="#FFFFFF", border=BORDER, title_size=8.2, sub_size=6.7)

    draw_arrow(ax, (28, 77), (28, 71.5), color=PRIMARY)
    draw_arrow(ax, (72, 77), (72, 71.5), color=SECONDARY)

    # Clean intermediate distribution bar
    dist_bar = FancyBboxPatch(
        (10, 56.5), 80, 3.2,
        boxstyle="round,pad=0.1,rounding_size=0.8",
        linewidth=1.0, edgecolor="#94A3B8", facecolor="#F1F5F9"
    )
    ax.add_patch(dist_bar)
    ax.text(50, 58.1, "Cleaned Normalized Texts (CV Text & JD Text) feed both parallel analytical streams",
            ha="center", va="center", fontsize=7.2, weight="bold", color=SLATE)

    draw_arrow(ax, (28, 63), (28, 59.7), color=PRIMARY)
    draw_arrow(ax, (72, 63), (72, 59.7), color=SECONDARY)

    # 2. Parallel Dual Streams
    # Container Left: Stream A (Rule-Based Skill Extraction)
    box_a = FancyBboxPatch(
        (4, 27.5), 43, 25.5,
        boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=1.2, edgecolor=PRIMARY_BORDER, facecolor=PRIMARY_LIGHT
    )
    ax.add_patch(box_a)
    draw_badge(ax, 6, 49.8, 39, 2.4, "STREAM A: SKILL EXTRACTION (40% WEIGHT)",
               bg=PRIMARY, text_color="#FFFFFF", size=7.2)

    draw_card(ax, 6, 39.5, 39, 9.0, "Regex Skill Extraction",
              "Matches CV & JD against 48+ predefined skills\nExtracts: Matched Skills & Missing Skills",
              bg="#FFFFFF", border=PRIMARY_BORDER, title_size=8.0, sub_size=6.6)
    draw_card(ax, 6, 29.2, 39, 8.5, "Skill Match Ratio Calculation",
              "Skill Ratio = (len(Matched Skills) / len(JD Skills)) × 100",
              bg="#FFFFFF", border=PRIMARY_BORDER, title_size=8.0, sub_size=6.6)
    draw_arrow(ax, (25.5, 39.5), (25.5, 37.7), color=PRIMARY)

    # Container Right: Stream B (Deep Semantic Representation)
    box_b = FancyBboxPatch(
        (53, 27.5), 43, 25.5,
        boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=1.2, edgecolor=SECONDARY_BORDER, facecolor=SECONDARY_LIGHT
    )
    ax.add_patch(box_b)
    draw_badge(ax, 55, 49.8, 39, 2.4, "STREAM B: SEMANTIC EMBEDDINGS (60% WEIGHT)",
               bg=SECONDARY, text_color="#FFFFFF", size=7.2)

    draw_card(ax, 55, 39.5, 39, 9.0, "SentenceTransformer Encoding",
              "paraphrase-multilingual-MiniLM-L12-v2 model\nGenerates 384-dimensional dense vectors for CV & JD",
              bg="#FFFFFF", border=SECONDARY_BORDER, title_size=8.0, sub_size=6.6)
    draw_card(ax, 55, 29.2, 39, 8.5, "Cosine Semantic Similarity",
              "cos_sim(CV_emb, JD_emb) × 100 (contextual alignment)",
              bg="#FFFFFF", border=SECONDARY_BORDER, title_size=8.0, sub_size=6.6)
    draw_arrow(ax, (74.5, 39.5), (74.5, 37.7), color=SECONDARY)

    # Arrows from distribution bar into Streams
    draw_arrow(ax, (25.5, 56.5), (25.5, 52.2), color=PRIMARY, lw=1.4)
    draw_arrow(ax, (74.5, 56.5), (74.5, 52.2), color=SECONDARY, lw=1.4)

    # 3. Fusion & Final Match Score (Bottom)
    final_box = FancyBboxPatch(
        (18, 5.0), 64, 18.5,
        boxstyle="round,pad=0.2,rounding_size=1.4",
        linewidth=1.4, edgecolor=ACCENT_BORDER, facecolor=ACCENT_LIGHT
    )
    ax.add_patch(final_box)

    draw_badge(ax, 28, 20.2, 44, 2.5, "WEIGHTED FUSION & FINAL MATCH SCORING",
               bg=ACCENT, text_color="#FFFFFF", size=7.5)

    ax.text(50, 16.2, "Final Score = (Semantic Similarity × 0.60) + (Skill Match Ratio × 0.40)",
            ha="center", va="center", fontsize=9.2, weight="bold", color=NAVY)

    # Clamping note & thresholds
    ax.text(50, 12.2, "Score clamped between 0.0% and 100.0% | Default to Semantic Score if JD has no predefined skills",
            ha="center", va="center", fontsize=7.2, color=SLATE)

    # Threshold tags
    draw_card(ax, 21, 6.8, 18, 4.2, ">= 75%: Strong Match", bg="#DCFCE7", border="#86EFAC",
              title_color="#166534", title_size=7.2)
    draw_card(ax, 41, 6.8, 18, 4.2, "50 - 74%: Moderate", bg="#FEF9C3", border="#FDE047",
              title_color="#854D0E", title_size=7.2)
    draw_card(ax, 61, 6.8, 18, 4.2, "< 50%: Low Match", bg="#FEE2E2", border="#FCA5A5",
              title_color="#991B1B", title_size=7.2)

    # Arrows leading to Fusion
    draw_arrow(ax, (25.5, 27.5), (38, 23.5), color=PRIMARY, lw=1.5)
    draw_arrow(ax, (74.5, 27.5), (62, 23.5), color=SECONDARY, lw=1.5)

    save_fig(fig, "figure_02_resume_analysis_process.png")


# ==============================================================================
# FIGURE 3: OVERALL APPLICATION WORKFLOW
# ==============================================================================
def overall_workflow():
    fig, ax = create_canvas(
        "Overall Application Workflow",
        "End-to-end user journey across the three Gradio operational modes",
        figsize=(13, 8.5)
    )

    # Entry Point
    draw_card(ax, 34, 79, 32, 8.5, "LAUNCH APPLICATION",
              "python app.py  →  Gradio Web UI (127.0.0.1:7860)",
              bg=PRIMARY_LIGHT, border=PRIMARY_BORDER, title_color=PRIMARY)

    draw_card(ax, 34, 66, 32, 8.5, "SELECT MODE TAB",
              "User chooses intended task from the Gradio navigation header",
              bg=ACCENT_LIGHT, border=ACCENT_BORDER, title_color=ACCENT)

    draw_arrow(ax, (50, 79), (50, 74.5), color=PRIMARY)

    # 3 Parallel Operational Modes
    modes = [
        {
            "x": 3.5, "color": PRIMARY, "bg": PRIMARY_LIGHT, "border": PRIMARY_BORDER,
            "badge": "MODE 1: RESUME ANALYZER",
            "inputs": ("Single Candidate Inputs", "1 CV (Pasted text or PDF) + 1 Job Description"),
            "processing": ("Core NLP Engine", "Dual-stream scoring, ATS check, Fake JD detection"),
            "outputs": ("Diagnostic Outputs", "Match %, Skill Gap Pie Chart, Keyword Frequency,\nATS Checklist & Level-based Mock Interview Questions")
        },
        {
            "x": 35.5, "color": SECONDARY, "bg": SECONDARY_LIGHT, "border": SECONDARY_BORDER,
            "badge": "MODE 2: COVER LETTER",
            "inputs": ("Applicant Inputs", "CV text, Job Description + Optional focus skills"),
            "processing": ("Template Synthesis", "Merges matched skills with tailored professional structure"),
            "outputs": ("Generated Deliverable", "Full editable cover letter draft aligned\nwith candidate strengths and JD requirements")
        },
        {
            "x": 67.5, "color": PURPLE, "bg": PURPLE_LIGHT, "border": PURPLE_BORDER,
            "badge": "MODE 3: RECRUITER MODE",
            "inputs": ("Batch Hiring Inputs", "Multiple PDF Resumes + 1 Target Job Description"),
            "processing": ("Batch Ranking Pipeline", "Parallel resume parsing, extraction & blended scoring"),
            "outputs": ("Comparative Analytics", "Ranked candidate comparison table +\nVisual bar chart highlighting top matched candidates")
        }
    ]

    for m in modes:
        x = m["x"]
        w = 29

        # Column background container
        col_box = FancyBboxPatch(
            (x, 6), w, 56,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            linewidth=1.2, edgecolor=m["border"], facecolor=m["bg"]
        )
        ax.add_patch(col_box)

        # Mode Badge
        draw_badge(ax, x + 1.5, 58.5, w - 3, 2.8, m["badge"],
                   bg=m["color"], text_color="#FFFFFF", size=7.2)

        # Inputs Card
        draw_card(ax, x + 1.5, 44.5, w - 3, 11, m["inputs"][0], m["inputs"][1],
                  bg="#FFFFFF", border=m["border"], title_color=NAVY, title_size=7.8, sub_size=6.6)

        # Processing Card
        draw_card(ax, x + 1.5, 27.5, w - 3, 11, m["processing"][0], m["processing"][1],
                  bg="#FFFFFF", border=m["border"], title_color=NAVY, title_size=7.8, sub_size=6.6)

        # Outputs Card
        draw_card(ax, x + 1.5, 8.5, w - 3, 13.5, m["outputs"][0], m["outputs"][1],
                  bg="#FFFFFF", border=m["border"], title_color=NAVY, title_size=7.8, sub_size=6.5)

        # Column inner arrows
        draw_arrow(ax, (x + w/2, 44.5), (x + w/2, 38.5), color=m["color"])
        draw_arrow(ax, (x + w/2, 27.5), (x + w/2, 22), color=m["color"])

    # Branching arrows from Mode Selector to Columns
    draw_arrow(ax, (50, 66), (18, 61.5), color=PRIMARY, rad=0.08)
    draw_arrow(ax, (50, 66), (50, 61.5), color=SECONDARY)
    draw_arrow(ax, (50, 66), (82, 61.5), color=PURPLE, rad=-0.08)

    save_fig(fig, "figure_03_overall_workflow.png")


# ==============================================================================
# FIGURE 4: USE CASE DIAGRAM (STANDARD CLEAN UML, NO CROSSING LINES)
# ==============================================================================
def use_case():
    fig, ax = create_canvas(
        "Use Case Diagram",
        "Actor interactions and functional boundaries of the Resume Optimization System",
        figsize=(13, 8.5)
    )

    # System boundary box (UML standard)
    boundary = FancyBboxPatch(
        (28, 8), 68, 78,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.4, edgecolor=PRIMARY, facecolor="#FAFCFF"
    )
    ax.add_patch(boundary)

    # System boundary header
    ax.text(62, 82.5, "System Boundary: Resume Optimization System",
            ha="center", va="center", fontsize=11, weight="bold", color=PRIMARY)

    # Actor drawing function
    def draw_actor(x, y, label, role_desc):
        # Head (solid circle with clean ring)
        ax.add_patch(Circle((x, y + 4.8), 2.0, facecolor="#EFF6FF", edgecolor=NAVY, linewidth=1.5))
        # Torso
        ax.plot([x, x], [y + 2.8, y - 3.2], color=NAVY, lw=1.6)
        # Arms
        ax.plot([x - 3.5, x + 3.5], [y + 0.5, y + 0.5], color=NAVY, lw=1.6)
        # Legs
        ax.plot([x, x - 2.8], [y - 3.2, y - 8.5], color=NAVY, lw=1.6)
        ax.plot([x, x + 2.8], [y - 3.2, y - 8.5], color=NAVY, lw=1.6)
        # Label & subtitle
        ax.text(x, y - 11.2, label, ha="center", fontsize=9.2, weight="bold", color=NAVY)
        ax.text(x, y - 13.8, role_desc, ha="center", fontsize=7.2, color=MUTED)

    draw_actor(12, 59, "Job Seeker", "Individual Applicant")
    draw_actor(12, 23, "Recruiter", "Hiring Professional")

    # Use Cases for Job Seeker (Top zone inside boundary, cleanly separated)
    job_seeker_cases = [
        (48, 73.5, "Analyze Single Resume & JD", "Evaluate skill match and text alignment"),
        (78, 73.5, "View Blended Match Score", "Quantitative % score and level rating"),
        (48, 59.5, "Inspect Skill Gaps & Keywords", "Matched/missing skills & frequency chart"),
        (78, 59.5, "Verify ATS & Job Authenticity", "Formatting checklist and red-flag alerts"),
        (48, 45.5, "Generate Tailored Cover Letter", "Automated draft incorporating key skills"),
        (78, 45.5, "Practice Interview Questions", "Dynamic Easy, Medium & Hard prep questions"),
    ]

    # Use Cases for Recruiter (Bottom zone inside boundary)
    recruiter_cases = [
        (48, 25.5, "Batch Upload Multiple PDF CVs", "Process candidate pool against single JD"),
        (78, 25.5, "Compare Candidate Match Scores", "Visual bar chart highlighting top candidates"),
        (48, 14.5, "Review Candidate Ranking Table", "Sortable table with details and match status"),
        (78, 14.5, "Identify Top Talent Pool", "Filter and select top qualified applicants"),
    ]

    def draw_use_case_oval(x, y, title, subtitle, color=SECONDARY, bg=SECONDARY_LIGHT, border=SECONDARY_BORDER):
        # Draw clean pill-shaped use case node
        w, h = 26, 8.5
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.2,rounding_size=3.5",
            linewidth=1.2, edgecolor=border, facecolor=bg
        )
        ax.add_patch(patch)
        ax.text(x, y + 1.2, title, ha="center", va="center", fontsize=7.8, weight="bold", color=NAVY)
        ax.text(x, y - 1.8, subtitle, ha="center", va="center", fontsize=6.4, color=SLATE)

    for x, y, title, sub in job_seeker_cases:
        draw_use_case_oval(x, y, title, sub, color=PRIMARY, bg=PRIMARY_LIGHT, border=PRIMARY_BORDER)

    for x, y, title, sub in recruiter_cases:
        draw_use_case_oval(x, y, title, sub, color=PURPLE, bg=PURPLE_LIGHT, border=PURPLE_BORDER)

    # Actor association lines (UML: Solid line without arrowheads from Actor anchor to Use Case oval)
    js_anchor = (15.5, 59)
    # Direct connections to left-column use cases of Job Seeker (NEVER crossing over any other oval!)
    ax.plot([js_anchor[0], 35], [js_anchor[1], 73.5], color=PRIMARY, lw=1.2)
    ax.plot([js_anchor[0], 35], [js_anchor[1], 59.5], color=PRIMARY, lw=1.2)
    ax.plot([js_anchor[0], 35], [js_anchor[1], 45.5], color=PRIMARY, lw=1.2)

    # Clean internal <<include>> / <<extend>> relationships between left & right use cases
    def draw_uml_include(start, end, label="<<include>>"):
        draw_arrow(ax, start, end, color=MUTED, lw=1.1, head_size=9, dashed=True)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 1.2
        ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=6.2, color=SLATE,
                bbox=dict(boxstyle="square,pad=0.1", fc="#FAFCFF", ec="none"))

    draw_uml_include((61, 73.5), (65, 73.5))
    draw_uml_include((61, 59.5), (65, 59.5))
    draw_uml_include((61, 45.5), (65, 45.5), label="<<extend>>")

    # Recruiter anchor point at (15.5, 23)
    rec_anchor = (15.5, 23)
    ax.plot([rec_anchor[0], 35], [rec_anchor[1], 25.5], color=PURPLE, lw=1.2)
    ax.plot([rec_anchor[0], 35], [rec_anchor[1], 14.5], color=PURPLE, lw=1.2)

    draw_uml_include((61, 25.5), (65, 25.5))
    draw_uml_include((61, 14.5), (65, 14.5))

    # Caption / Legend at the bottom
    legend_text = "Solid lines indicate primary actor participation. Dashed arrows denote UML <<include>> or <<extend>> dependencies."
    ax.text(50, 4.2, legend_text, ha="center", va="center", fontsize=7.2, color=MUTED, style="italic")

    save_fig(fig, "figure_04_use_case_diagram.png")


if __name__ == "__main__":
    architecture()
    analysis_process()
    overall_workflow()
    use_case()
    print("\nAll four publication-quality figures generated successfully!")
    print(f"Output directory: {OUT.resolve()}")
