import os
import re
from functools import lru_cache
from typing import Dict, List

from sentence_transformers import SentenceTransformer, util

# --- Environment Setup ---
# Force offline mode by default for fast, reproducible local execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Predefined skill dictionary for keyword extraction
SKILLS: List[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql",
    "html", "css", "react", "node.js", "django", "flask", "fastapi",
    "machine learning", "ml", "deep learning", "nlp", "computer vision",
    "data analysis", "data analytics", "data science", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "keras",
    "power bi", "tableau", "excel", "git", "github", "version control",
    "aws", "azure", "gcp", "docker", "kubernetes", "linux",
    "communication", "teamwork", "leadership", "problem solving",
    "project management", "agile", "scrum",
]


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """
    Load and cache the pre-trained SentenceTransformer model.
    Using lru_cache ensures the heavy model is only loaded into memory once, 
    significantly improving the response time for subsequent requests.

    Attempts to load from the local cache first (offline mode).
    If the model is not found in cache, it attempts to download it automatically
    if internet access is available, or provides a clear error message.
    """
    try:
        # First attempt: load from local cache (strictly offline)
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        # Fallback: if not cached, temporarily allow network access to download
        try:
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            model = SentenceTransformer(MODEL_NAME, local_files_only=False)
            # Re-enable offline mode after successful download
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            return model
        except Exception as e:
            raise RuntimeError(
                f"\n\n[Model Loading Error: '{MODEL_NAME}']\n"
                "The model was not found in your local Hugging Face cache and could not be downloaded.\n"
                "Please connect to the internet and run this command once:\n"
                f'  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'{MODEL_NAME}\')"\n\n'
                "See README.md (Step 3: Download the model) for details.\n"
            ) from e


def normalize_text(text: str) -> str:
    """Converts text to lowercase and removes extra whitespaces for consistency."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def extract_skills(text: str) -> List[str]:
    """
    Extracts known skills from the given text using regular expressions.
    Returns a sorted list of unique skills found in the text.
    """
    text_norm = normalize_text(text)
    found_skills = []
    
    for skill in SKILLS:
        # Regex ensures exact word matches (e.g., 'java' won't match 'javascript')
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_norm):
            found_skills.append(skill)
            
    return sorted(set(found_skills))


def extract_text_from_pdf(path: str) -> str:
    """Extracts raw text content from an uploaded PDF file."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
        
    reader = PdfReader(path)
    # Combine text from all pages into a single string
    text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    
    if not text:
        raise ValueError("Could not extract text from the PDF. It might be an image-based PDF.")
    return text


def analyze_match(cv_text: str, jd_text: str) -> Dict:
    """
    Compares the CV and Job Description to calculate a reliable match score.
    Uses a blended metric: 60% Semantic Similarity + 40% Skill Match Ratio.
    """
    cv_text = (cv_text or "").strip()
    jd_text = (jd_text or "").strip()
    
    if not cv_text or not jd_text:
        raise ValueError("Both CV and Job Description are required.")

    model = load_model()
    
    # 1. Semantic Similarity (Contextual meaning)
    # Encode both texts into dense vectors and calculate cosine similarity
    cv_emb = model.encode(cv_text, convert_to_tensor=True, normalize_embeddings=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
    raw_similarity = float(util.cos_sim(cv_emb, jd_emb)[0][0])
    semantic_score = raw_similarity * 100

    # 2. Skill Match Ratio (Exact keyword matches)
    cv_skills = set(extract_skills(cv_text))
    jd_skills = set(extract_skills(jd_text))
    
    matched_skills = sorted(cv_skills & jd_skills)
    missing_skills = sorted(jd_skills - cv_skills)
    
    # Avoid division by zero if the JD does not contain any recognizable skills
    if len(jd_skills) > 0:
        skill_match_ratio = (len(matched_skills) / len(jd_skills)) * 100
    else:
        skill_match_ratio = semantic_score 

    # 3. Blended Final Score
    # This blended approach ensures the score is realistic and penalizes missing skills
    final_score = (semantic_score * 0.6) + (skill_match_ratio * 0.4)
    score = round(min(100.0, max(0.0, final_score)), 1)

    # Determine the qualitative match level
    if score >= 75:
        level = "Strong match"
    elif score >= 50:
        level = "Moderate match"
    else:
        level = "Low match"

    return {
        "score": score,
        "similarity": raw_similarity,
        "level": level,
        "matched": matched_skills,
        "missing": missing_skills
    }
