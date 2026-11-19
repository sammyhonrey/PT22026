import re
import os
from collections import Counter
from typing import List
import matplotlib.pyplot as plt

# Common English stopwords to filter out during keyword extraction
STOPWORDS = set([
    "the", "and", "to", "of", "in", "a", "is", "for", "with", "on", "that",
    "this", "an", "as", "are", "be", "by", "or", "from", "at", "it", "we",
    "you", "will", "your", "our", "have", "has", "must", "should", "can"
])

def get_filtered_words(text: str) -> List[str]:
    """Extracts words with 3 or more letters and removes common stopwords."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

def extract_top_keywords(text: str, top_n: int = 8) -> str:
    """Returns a formatted markdown string of the most frequent keywords in the JD."""
    top_words = Counter(get_filtered_words(text)).most_common(top_n)
    if not top_words:
        return "No significant keywords found."
    result = "### Top Keywords in JD\n"
    for word, count in top_words:
        result += f"- **{word.capitalize()}** (appears {count} times)\n"
    return result

def create_keyword_chart(text: str):
    """Generates a dark-themed bar chart for the top keywords in the JD."""
    top_words = Counter(get_filtered_words(text)).most_common(7)
    if not top_words:
        return None
    labels = [word[0].capitalize() for word in top_words]
    counts = [word[1] for word in top_words]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    ax.bar(labels, counts, color='#3b82f6', width=0.5)

    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='#cbd5e1', labelsize=10)
    ax.tick_params(axis='y', colors='#cbd5e1', labelsize=10)
    ax.set_title("Top Keywords Frequency", color='#f8fafc', pad=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_skill_gap_chart(matched: List[str], missing: List[str]):
    """Generates a pie chart visually showing the ratio of matched vs missing skills."""
    if not matched and not missing:
        return None
    labels = ['Matched Skills', 'Missing Skills']
    sizes = [len(matched), len(missing)]
    colors = ['#10b981', '#ef4444']

    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
        startangle=90, textprops={'color': '#cbd5e1', 'fontweight': 'bold'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
    ax.axis('equal')
    ax.set_title("Skill Gap Analysis", color='#f8fafc', pad=15, fontweight='bold')
    plt.tight_layout()
    return fig

def create_comparison_chart(candidate_names: List[str], scores: List[float], top_n: int = 3):
    """
    Generates a bar chart comparing multiple candidates.
    The top N candidates are colored green to highlight them as the best matches.
    """
    if not candidate_names or not scores:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    # Color the top N bars green, and the rest blue
    colors = ['#10b981' if i < top_n else '#3b82f6' for i in range(len(candidate_names))]
    bars = ax.bar(candidate_names, scores, color=colors, width=0.5)

    # Add percentage labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval}%', ha='center', va='bottom', color='#cbd5e1', fontweight='bold')

    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='#cbd5e1', labelsize=10)
    ax.tick_params(axis='y', colors='#cbd5e1', labelsize=10)
    ax.set_ylim(0, 110)
    ax.set_title(f"Candidate Match Score Comparison (Top {top_n} Highlighted)", color='#f8fafc', pad=15, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return fig

# Interview question bank with difficulty levels.
# Each skill has one easy, one medium and one hard question.
INTERVIEW_QUESTION_BANK = {
    "python": {
        "easy": "What is the difference between a list and a tuple in Python?",
        "medium": "How does Python handle memory management and garbage collection?",
        "hard": "Explain the GIL and its impact on multi-threaded Python programs."
    },
    "sql": {
        "easy": "What is the difference between INNER JOIN and LEFT JOIN?",
        "medium": "What is normalization? Explain 1NF, 2NF and 3NF with examples.",
        "hard": "A query on a table with millions of rows is slow. How would you find the cause and optimize it?"
    },
    "machine learning": {
        "easy": "What is the difference between supervised and unsupervised learning?",
        "medium": "How do you handle overfitting in machine learning models?",
        "hard": "Explain the bias-variance tradeoff and how you would diagnose it in a real project."
    },
    "docker": {
        "easy": "What is the difference between a Docker image and a container?",
        "medium": "What is a Dockerfile and how do image layers work?",
        "hard": "How would you reduce the size of a Docker image for production deployment?"
    },
    "git": {
        "easy": "What is the difference between git pull and git fetch?",
        "medium": "How do you resolve a merge conflict in Git?",
        "hard": "Explain the difference between git merge and git rebase, and when you would use each."
    },
    "aws": {
        "easy": "What is the difference between EC2 and S3 in AWS?",
        "medium": "What is an IAM role and why is it important for security?",
        "hard": "How would you design a scalable architecture for a high-traffic web application on AWS?"
    },
    "react": {
        "easy": "What is the difference between state and props in React?",
        "medium": "What are React hooks and why were they introduced?",
        "hard": "How does React's reconciliation and diffing process work under the hood?"
    },
    "django": {
        "easy": "What is the MVT architecture in Django?",
        "medium": "What is the purpose of middleware in Django?",
        "hard": "How do you optimize Django ORM queries, and what is the difference between select_related and prefetch_related?"
    },
    "javascript": {
        "easy": "What is the difference between var, let and const?",
        "medium": "Can you explain closures in JavaScript with an example?",
        "hard": "Explain how the JavaScript event loop works, including promises and async/await."
    },
    "java": {
        "easy": "What are the four main principles of OOP?",
        "medium": "What is the difference between an interface and an abstract class in Java?",
        "hard": "How does JVM garbage collection work and how would you tune it?"
    },
    "html": {
        "easy": "What are semantic HTML tags and why should you use them?",
        "medium": "How do you make a web page accessible to screen readers?",
        "hard": "How does the browser render a page from HTML and CSS (the critical rendering path)?"
    },
    "css": {
        "easy": "What is the CSS box model?",
        "medium": "When would you use Flexbox versus CSS Grid?",
        "hard": "Explain CSS specificity and how the cascade resolves conflicting styles."
    },
    "pandas": {
        "easy": "What is the difference between a Series and a DataFrame in pandas?",
        "medium": "How does groupby work and when would you use it?",
        "hard": "What is the difference between merge, join and concat, and what are their performance tradeoffs?"
    },
    "numpy": {
        "easy": "What is the difference between a NumPy array and a Python list?",
        "medium": "What is broadcasting in NumPy?",
        "hard": "Why is vectorization faster than loops in NumPy? Explain at the memory level."
    },
    "excel": {
        "easy": "What is the difference between VLOOKUP and INDEX-MATCH?",
        "medium": "What are pivot tables and when would you use one?",
        "hard": "How would you clean and validate a large messy dataset in Excel?"
    },
}

# Aliases that should share the same question bank entry
SKILL_ALIASES = {
    "ml": "machine learning",
    "js": "javascript",
}

# Generic fallback questions for skills that are not in the bank,
# still split by difficulty so every skill gets a real practice path.
GENERIC_QUESTIONS = {
    "easy": "What are the core concepts of {skill} that every developer should know?",
    "medium": "Describe a challenging project where you used {skill}. What problems did you face and how did you solve them?",
    "hard": "What are the limitations or common pitfalls of {skill}, and how do you work around them in production?"
}

def generate_interview_questions(matched_skills: List[str], max_skills: int = 6) -> str:
    """
    Generates mock interview questions grouped by difficulty (easy, medium, hard).
    Each matched skill gets one question per level, from a curated question bank
    when available, otherwise from generic but level-specific fallbacks.
    """
    if not matched_skills:
        return "### Mock Interview Questions\nNo specific technical skills matched."

    # Normalize skill names, resolve aliases and keep the original order
    normalized = []
    seen = set()
    for skill in matched_skills:
        key = skill.lower().strip()
        key = SKILL_ALIASES.get(key, key)
        if key and key not in seen:
            seen.add(key)
            normalized.append((skill, key))

    # Limit the number of skills so the output stays readable
    normalized = normalized[:max_skills]

    levels = [
        ("easy", "Easy (Warm-up Questions)"),
        ("medium", "Medium (Core Knowledge)"),
        ("hard", "Hard (Deep Dive)"),
    ]

    result = "### Mock Interview Questions\nBased on your matched skills, practice from easy to hard:\n"
    for level_key, level_title in levels:
        result += f"\n**{level_title}**\n"
        for display_name, key in normalized:
            bank_entry = INTERVIEW_QUESTION_BANK.get(key)
            if bank_entry:
                question = bank_entry[level_key]
            else:
                question = GENERIC_QUESTIONS[level_key].format(skill=display_name.capitalize())
            result += f"- **{display_name.capitalize()}**: {question}\n"
    return result

def detect_fake_job(jd_text: str) -> str:
    jd_lower = jd_text.lower()
    red_flags = []
    scam_keywords = {
        "registration fee": "Legitimate companies never ask for money for registration.",
        "pay a fee": "Never pay money to get a job.",
        "deposit": "Asking for a deposit is a common scam tactic.",
        "whatsapp": "Interviews conducted purely via WhatsApp are highly suspicious.",
        "telegram": "Recruiters using Telegram for hiring are often scammers.",
        "no experience needed": "Unrealistic promises like 'no experience, high pay' are red flags.",
        "easy money": "Too-good-to-be-true promises are usually scams.",
        "earn $": "High daily/weekly earnings for basic work is a major red flag.",
        "western union": "Scammers often use untraceable payment methods.",
        "bitcoin": "Crypto payments for jobs are a common scam.",
    }
    for keyword, reason in scam_keywords.items():
        if keyword in jd_lower:
            red_flags.append(f"- **Detected '{keyword}'**: {reason}")

    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', jd_text)
    for email in emails:
        if "@gmail.com" in email or "@yahoo.com" in email or "@hotmail.com" in email:
            red_flags.append(f"- **Suspicious Email ({email})**: Legitimate companies use corporate email addresses.")

    if not red_flags:
        return "### Job Authenticity Check\nLooks Safe: No common scam patterns detected."

    result = "### Job Authenticity Check\nWarning: Potential Red Flags Detected\n\n"
    result += "\n".join(red_flags)
    result += "\n\n*Note: Always verify the company independently.*"
    return result

def check_ats_compatibility(cv_text: str, missing_skills: List[str], matched_skills: List[str]) -> str:
    score = 100
    issues = []
    if not re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text):
        score -= 15
        issues.append("- **Missing Email**: ATS requires a valid email.")
    if not re.search(r'\+?\d[\d -]{8,12}\d', cv_text):
        score -= 15
        issues.append("- **Missing Phone Number**: Add a contact number.")
    word_count = len(cv_text.split())
    if word_count < 100:
        score -= 10
        issues.append(f"- **Too Short**: {word_count} words. Add more details.")
    elif word_count > 1200:
        score -= 10
        issues.append("- **Too Long**: Keep it concise.")
    if missing_skills:
        total_jd_skills = len(missing_skills) + len(matched_skills)
        if total_jd_skills > 0 and (len(missing_skills) / total_jd_skills) > 0.6:
            score -= 10
            issues.append("- **Low Keyword Density**: Missing many required keywords.")
    final_score = max(0, score)
    if not issues:
        return f"### ATS Score: {final_score}/100 (Pass)\nYour CV passes basic ATS checks."
    issues_str = "\n".join(issues)
    return f"### ATS Score: {final_score}/100\n**Issues found:**\n{issues_str}"

def generate_feedback(score: float, missing_skills: List[str]) -> str:
    if score >= 75:
        return "Excellent match! Your CV aligns very well with the job description."
    elif score >= 50:
        feedback = "Moderate match. Your CV covers some key areas, but you should strengthen it."
        if missing_skills:
            feedback += f" Consider adding these missing skills: {', '.join(missing_skills)}."
        return feedback
    else:
        return "Low match. Your CV does not seem to meet the core requirements."

def get_learning_recommendations(missing_skills: List[str]) -> str:
    if not missing_skills:
        return "No missing skills detected. Your CV looks strong!"
    resources = {
        "python": "Try 'Python for Everybody' on Coursera.",
        "sql": "Practice on LeetCode Database or W3Schools.",
        "machine learning": "Take Andrew Ng's ML Specialization on Coursera.",
        "ml": "Take Andrew Ng's ML Specialization on Coursera.",
        "docker": "Read Docker's official 'Get Started' guide.",
        "git": "Check GitHub's Git Handbook.",
        "data analysis": "Google Data Analytics Certificate on Coursera.",
        "data science": "Google Data Analytics Certificate on Coursera.",
        "communication": "Practice with Toastmasters.",
        "aws": "AWS Cloud Practitioner Essentials.",
        "react": "Follow the official React tutorial (react.dev).",
    }
    recs = ["Here are recommended resources:"]
    for skill in missing_skills:
        res = resources.get(skill, "Search for free tutorials on YouTube.")
        recs.append(f"- **{skill.capitalize()}**: {res}")
    return "\n".join(recs)


# ---------------------------------------------------------------------------
# Cover letter generation helpers
# ---------------------------------------------------------------------------

_SECTION_ALIASES = {
    "education": ["education", "academic background", "academics", "qualifications"],
    "skills": [
        "skills", "technical skills", "core competencies", "technologies",
        "skills technologies", "technical proficiencies"
    ],
    "experience": [
        "work experience", "professional experience", "experience",
        "employment history", "employment", "work history"
    ],
    "projects": [
        "projects", "personal projects", "academic projects",
        "selected projects", "key projects"
    ],
    "achievements": [
        "achievements", "awards", "honors", "honours",
        "certifications", "certificates", "achievements certifications"
    ],
}

_NAME_EXCLUDED_WORDS = {
    "resume", "curriculum", "vitae", "summary", "objective", "profile",
    "contact", "phone", "email", "address", "linkedin", "github",
    "education", "experience", "skills", "projects", "achievements",
    "engineer", "developer", "manager", "analyst", "scientist", "designer",
    "consultant", "intern", "specialist", "architect", "administrator",
    "director", "lead", "senior", "junior", "software", "data", "marketing",
    "sales", "professional", "about",
}


def _clean_line(line: str) -> str:
    """Normalize a resume/JD line and remove common leading bullet characters."""
    return re.sub(r"\s+", " ", (line or "").strip(" \t\r\n-•*")).strip()


def _normalized_heading(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").lower()).strip(" \t-•*")


def _is_section_heading(line: str, aliases: List[str]) -> bool:
    normalized = _normalized_heading(line)
    for alias in aliases:
        alias = alias.lower()
        if (
            normalized == alias
            or normalized.startswith(alias + ":")
            or normalized.startswith(alias + " -")
            or normalized.startswith(alias + " –")
        ):
            return True
    return False


def _is_any_section_heading(line: str) -> bool:
    return any(_is_section_heading(line, aliases) for aliases in _SECTION_ALIASES.values())


def _extract_section_lines(text: str, aliases: List[str], max_lines: int = 4) -> List[str]:
    """Extract lines under a section heading, stopping at the next known section."""
    lines = (text or "").splitlines()
    start_index = None
    inline_content = ""

    for index, line in enumerate(lines):
        if _is_section_heading(line, aliases):
            start_index = index + 1
            if ":" in line:
                remainder = line.split(":", 1)[1].strip()
                if remainder:
                    inline_content = _clean_line(remainder)
            break

    if start_index is None:
        return []

    collected = []
    if inline_content:
        collected.append(inline_content)

    for line in lines[start_index:]:
        if _is_any_section_heading(line):
            break
        cleaned = _clean_line(line)
        if cleaned:
            collected.append(cleaned)
        if len(collected) >= max_lines:
            break

    return collected[:max_lines]


def _format_person_name(name: str) -> str:
    name = _clean_line(name).strip(" ,|")
    if name.isupper():
        name = name.title()
    return name


def _looks_like_person_name(line: str) -> bool:
    cleaned = _clean_line(line)
    if not cleaned or _is_any_section_heading(cleaned):
        return False
    if re.search(r"[@\d]|https?://|www\.", cleaned):
        return False
    if len(cleaned) > 60:
        return False

    words = cleaned.split()
    if len(words) == 1:
        return (
            cleaned.isupper()
            and len(cleaned) >= 3
            and cleaned.lower() not in _NAME_EXCLUDED_WORDS
        )

    if not (2 <= len(words) <= 5):
        return False

    lowered = {word.lower().strip(".,") for word in words}
    if lowered & _NAME_EXCLUDED_WORDS:
        return False

    return all(
        re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word.strip(".,"))
        for word in words
    )


def _extract_candidate_name(cv_text: str) -> str:
    cleaned_lines = [
        _clean_line(line)
        for line in (cv_text or "").splitlines()
        if _clean_line(line)
    ]

    for line in cleaned_lines[:15]:
        match = re.match(r"(?i)^name\s*[:\-]\s*(.+)$", line)
        if match:
            return _format_person_name(match.group(1))

    for line in cleaned_lines[:8]:
        # Prefer the part before a pipe or comma, e.g. "Jane Doe | Software Engineer"
        candidate = re.split(r"\s*[|]\s*", line)[0].strip()
        if _looks_like_person_name(candidate):
            return _format_person_name(candidate)

        candidate = re.split(r"\s*,\s*", line)[0].strip()
        if _looks_like_person_name(candidate):
            return _format_person_name(candidate)

    return ""


def _extract_job_info(jd_text: str):
    """Extract a likely job title and company from the JD, only when explicitly available."""
    lines = [
        _clean_line(line)
        for line in (jd_text or "").splitlines()
        if _clean_line(line)
    ]

    title = ""
    company = ""

    for line in lines[:20]:
        match = re.match(r"(?i)^(?:job\s*title|position|role)\s*[:\-]\s*(.+)$", line)
        if match:
            title = _clean_line(match.group(1))
            break

    if not title:
        for line in lines[:8]:
            match = re.search(
                r"(?i)(?:seeking|looking for|hiring|recruiting)\s+(?:a|an)\s+"
                r"([A-Za-z0-9][A-Za-z0-9 /+.-]{2,70})",
                line,
            )
            if match:
                title = _clean_line(match.group(1))
                break

    if not title and lines:
        first = lines[0]
        if (
            len(first) <= 80
            and not re.search(
                r"(?i)\b(job description|about|we are|our company|company overview|responsibilities)\b",
                first,
            )
            and not first.endswith(".")
        ):
            title = first

    for line in lines[:20]:
        match = re.match(r"(?i)^(?:company|organization|employer)\s*[:\-]\s*(.+)$", line)
        if match:
            company = _clean_line(match.group(1))
            break

    if not company:
        for line in lines[:8]:
            match = re.search(
                r"(?i)\b(?:join|at)\s+([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+){0,3})",
                line,
            )
            if match:
                company = _clean_line(match.group(1))
                break

    if re.search(r"(?i)\s+at\s+", title):
        parts = re.split(r"\s+at\s+", title, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            title = parts[0].strip()
            if not company:
                company = parts[1].strip()

    title = re.sub(r"\s+", " ", title).strip(" .,:;-")
    company = re.sub(r"\s+", " ", company).strip(" .,:;-")
    return title, company


def _extract_education(cv_text: str) -> List[str]:
    lines = _extract_section_lines(cv_text, _SECTION_ALIASES["education"], max_lines=3)
    if lines:
        return lines[:2]

    fallback = []
    for line in (cv_text or "").splitlines():
        cleaned = _clean_line(line)
        if cleaned and re.search(
            r"(?i)\b(bachelor|master|b\.?s\.?|m\.?s\.?|ph\.?d|doctorate|university|college|institute|degree|diploma)\b",
            cleaned,
        ):
            fallback.append(cleaned)
            if len(fallback) >= 2:
                break
    return fallback


def _extract_experience(cv_text: str) -> List[str]:
    lines = _extract_section_lines(cv_text, _SECTION_ALIASES["experience"], max_lines=4)
    return lines[:2]


def _extract_projects(cv_text: str) -> List[str]:
    lines = _extract_section_lines(cv_text, _SECTION_ALIASES["projects"], max_lines=4)
    if lines:
        return lines[:2]

    fallback = []
    for line in (cv_text or "").splitlines():
        cleaned = _clean_line(line)
        if re.match(r"(?i)^project\s*[:\-]", cleaned):
            fallback.append(cleaned)
            if len(fallback) >= 2:
                break
    return fallback


def _extract_achievements(cv_text: str) -> List[str]:
    lines = _extract_section_lines(cv_text, _SECTION_ALIASES["achievements"], max_lines=3)
    if lines:
        return lines[:2]

    fallback = []
    for line in (cv_text or "").splitlines():
        cleaned = _clean_line(line)
        if re.search(r"(?i)\b(award|certified|certification|honor|honour|achievement|winner)\b", cleaned):
            fallback.append(cleaned)
            if len(fallback) >= 2:
                break
    return fallback


def _skill_is_supported(skill: str, cv_text: str) -> bool:
    skill = (skill or "").strip()
    if not skill:
        return False
    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
    return bool(re.search(pattern, cv_text, re.IGNORECASE))


def _filter_supported_skills(skills: List[str], cv_text: str) -> List[str]:
    """Keep only skills that appear in the resume text, preserving order and removing duplicates."""
    supported = []
    seen = set()
    for skill in skills or []:
        key = (skill or "").lower().strip()
        if key and key not in seen and _skill_is_supported(skill, cv_text):
            seen.add(key)
            supported.append(skill)
    return supported


def _human_list(items: List[str], conjunction: str = "and", sep: str = ", ") -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return f"{sep.join(items[:-1])}, {conjunction} {items[-1]}"


def _top_jd_keywords(jd_text: str, top_n: int = 3) -> List[str]:
    generic = {
        "team", "work", "role", "job", "company", "looking", "experience",
        "skills", "ability", "strong", "working", "years", "plus", "using",
        "knowledge", "requirements", "responsibilities", "candidate", "ideal",
        "new", "well", "also", "including", "within", "across", "related",
    }
    keywords = []
    for word, _count in Counter(get_filtered_words(jd_text)).most_common(25):
        if word in generic:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= top_n:
            break
    return keywords


def generate_cover_letter(cv_text: str, jd_text: str, matched_skills: List[str]) -> str:
    """
    Generate a dynamic cover letter from the resume, job description and supported skills.

    The function only uses information that is present in the resume. It does not invent
    education, experience, projects, achievements, or skills.
    """
    cv_text = (cv_text or "").strip()
    jd_text = (jd_text or "").strip()

    if not cv_text:
        return "Please provide your resume text."
    if not jd_text:
        return "Please provide the job description."

    name = _extract_candidate_name(cv_text)
    job_title, company = _extract_job_info(jd_text)
    supported_skills = _filter_supported_skills(matched_skills or [], cv_text)
    education = _extract_education(cv_text)
    experience = _extract_experience(cv_text)
    projects = _extract_projects(cv_text)
    achievements = _extract_achievements(cv_text)
    jd_keywords = _top_jd_keywords(jd_text)

    greeting = "Dear Hiring Manager,"
    paragraphs = []

    position_phrase = (
        f"the {job_title} position" if job_title else "the position described in your job posting"
    )
    company_phrase = f" at {company}" if company else ""

    if name:
        opening = (
            f"My name is {name}, and I am writing to express my interest in "
            f"{position_phrase}{company_phrase}."
        )
    else:
        opening = (
            f"I am writing to express my interest in "
            f"{position_phrase}{company_phrase}."
        )

    if jd_keywords:
        opening += f" I was particularly drawn to the role’s focus on {_human_list(jd_keywords)}."

    paragraphs.append(opening)

    if supported_skills:
        skill_names = [skill.capitalize() for skill in supported_skills]
        skills_paragraph = (
            f"My resume demonstrates experience with {_human_list(skill_names)}."
        )
        if jd_keywords:
            skills_paragraph += (
                f" These capabilities align with the role’s emphasis on "
                f"{_human_list(jd_keywords)}."
            )
        paragraphs.append(skills_paragraph)

    if education:
        paragraphs.append(
            "My education and academic background include " + "; ".join(education) + "."
        )

    if experience:
        paragraphs.append(
            "Relevant experience from my resume includes: " + "; ".join(experience) + "."
        )

    if projects:
        paragraphs.append(
            "I have also worked on projects such as " + "; ".join(projects) + "."
        )

    if achievements:
        paragraphs.append(
            "Additional achievements and certifications from my resume include: "
            + "; ".join(achievements)
            + "."
        )

    paragraphs.append(
        "I would welcome the opportunity to discuss how my background and skills "
        "can contribute to your team. Thank you for your time and consideration."
    )

    closing = "Sincerely,\n" + (name if name else "[Your Name]")

    return greeting + "\n\n" + "\n\n".join(paragraphs) + "\n\n" + closing