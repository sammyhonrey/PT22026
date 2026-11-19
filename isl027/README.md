# Resume Optimization System

## About the Project

The Resume Optimization System is a Python application that compares a resume (CV) with a job description (JD). It uses a pre-trained SentenceTransformer model to compare the meaning of both texts. It also checks for skills that appear in the job description and whether those skills are included in the resume.

I built the application with Gradio so users can work with it through a web interface. It is designed to help job seekers review their resumes, identify missing skills and prepare for job applications. Recruiters can also use it to compare several resumes against the same job description.

The match score is only a guide. It does not decide whether someone should be hired or guarantee an interview.

## Main Features

The application has three main sections.

### 1. Resume Analyzer

In this section, users can:

- Paste resume text or upload a text-based PDF.
- Compare a resume with a job description.
- View the match score and its category.
- Check which skills are matched and which are missing.
- View a chart showing the skill gap.
- See the most frequent keywords in the job description.
- Get basic ATS-related suggestions.
- Check a job description for selected warning signs.
- Receive sample interview questions and learning suggestions.
- Preview the resume with matched skills highlighted.

### 2. Cover Letter Generator

Users can enter their resume text and a job description. The application checks the skills in both texts and uses the matched skills to prepare a cover letter draft. Users can also add extra skills they want to mention.

The generated letter is a starting point, so it should be checked and edited before being sent to an employer.

### 3. Recruiter Mode

Recruiter Mode allows users to upload several text-based PDF resumes and compare them with one job description. The application shows the candidates in a ranking table and displays their match scores in a bar chart.

## Machine Learning Model and Scoring

The project uses this pre-trained model:

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Source:** Hugging Face Model Hub
- **Type:** SentenceTransformer
- **Purpose:** Text embeddings and semantic similarity

The model converts the resume and job description into numerical representations called embeddings. The application uses cosine similarity to measure how close the two texts are in meaning.

The final match score combines two parts:

- **60% semantic similarity**
- **40% skill match ratio**

The skill match ratio is based on the recognized skills in the job description that are also found in the resume. If no recognized skills are found in the job description, the semantic score is used for the skill component as well.

The final score is kept between 0 and 100.

## Match Categories

- **Strong match:** 75% or higher
- **Moderate match:** 50% to below 75%
- **Low match:** Below 50%

These ranges are defined for this project. They are not official hiring standards.

## Other Processing Methods

Not every feature uses a trained machine learning model. Some parts of the application use simple rules and predefined keyword lists:

- Skill extraction uses a fixed skill dictionary and regular expressions.
- ATS-related checks look for contact details, resume length and missing job skills.
- The job warning feature checks for selected suspicious phrases and email patterns.
- Interview questions and learning suggestions are selected using predefined rules.
- Keyword frequency is calculated by counting words in the job description.

These features provide basic guidance. They do not replace a professional ATS platform, a proper investigation of a job advertisement or a human review.

## Project Structure

```text
Resume-Optimization-System/
│
├── app.py
├── model_utils.py
├── nlp_features.py
├── evaluation.py
├── generate_report_figures.py
├── requirements.txt
├── README.md
├── evaluation_results.csv       # Created by evaluation.py
└── report_figures/              # Created by generate_report_figures.py
```

The `report_figures` folder is created when the figure-generation script runs. It contains the generated PNG and SVG files.

## Requirements

- Python 3.10 or a compatible newer version
- pip
- An internet connection for the first model download
- Enough memory to load the SentenceTransformer model

The required Python packages are listed in `requirements.txt`.

## Installation

Open a terminal in the project folder and follow these steps.

### 1. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell does not allow the environment to activate, run this command in the current terminal session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the required packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the model

The application is set to use offline mode when it starts. This means the pre-trained model needs to be downloaded to the local Hugging Face cache before running the application for the first time.

While connected to the internet, run:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

Wait for the download to finish. The application can then load the model from the local cache.

If the model has not been downloaded, the application will not be able to fetch it automatically because offline mode is enabled.

## Running the Application

After installing the packages and downloading the model, run:

```bash
python app.py
```

The application should open in your browser. If it does not, visit:

`http://127.0.0.1:7860`

## Running the Evaluation

The project includes an evaluation script with a small number of manually prepared test examples. To run it, use:

```bash
python evaluation.py
```

The script checks:

- Skill extraction against manually written expected results.
- Resume and job-description matching using strong, partial and low-match examples.
- The job warning feature using selected suspicious and normal job descriptions.
- Precision, recall and F1-score for skill extraction and the job warning tests.
- Match scores and categories for the candidate examples.

After the evaluation finishes, the script saves the detailed results in:

`evaluation_results.csv`

The test examples are limited and were prepared for this project. The results should not be treated as general performance measurements for all resumes or job advertisements. A larger test set with independently labelled examples would be needed for a more complete evaluation.

## Generating Figures for the Report

The project also includes a script for creating figures used in the technical report. Run:

```bash
python generate_report_figures.py
```

The script saves PNG and SVG figures in the `report_figures` folder. These figures are for the report and are not needed to run the main application.

Some diagrams, such as the use case diagram and detailed process flow, were prepared separately in Draw.io.

## Input Notes

- Resume text can be pasted directly into the application.
- PDF upload works with PDFs that contain selectable text.
- Scanned or image-only PDFs may not work because the project does not include OCR.
- Recruiter Mode accepts multiple text-based PDF resumes.
- A clear job description with specific role requirements will make the comparison more useful.

## Limitations

There are a few limitations to keep in mind:

1. **Fixed skill dictionary:** The application can only identify skills included in its predefined list. It may miss new or specialized skills.
2. **Exact skill matching:** The skill extractor looks for keyword patterns. It may not recognize every synonym or differently written version of a skill.
3. **Semantic similarity:** A high similarity score does not prove that a candidate is qualified for a job.
4. **Basic ATS checks:** The ATS score is based on a few simple checks. It is not a full ATS compatibility test and does not inspect every part of the resume layout.
5. **Job warning feature:** The detector looks for selected warning signs. It cannot confirm whether a job is real or fraudulent.
6. **Generated suggestions:** Cover letters, interview questions and learning recommendations may need editing or further checking.
7. **PDF extraction:** Scanned PDFs are not supported because OCR is not included.
8. **Evaluation data:** The evaluation uses a small set of manually prepared examples, not a large real-world benchmark.

## Acknowledgements

This project uses the following tools and resources:

- The pre-trained SentenceTransformer model from the Hugging Face Model Hub.
- Gradio for the web interface.
- Matplotlib for charts and figures.
- pypdf for extracting text from PDF resumes.
- pandas for organizing candidate comparison results.

## Responsible Use

This application is intended to support resume review, not to make hiring decisions. It should not be used as the only reason to hire, reject or rank a person. Job seekers and recruiters should always review the original resume and job requirements before making a decision.
