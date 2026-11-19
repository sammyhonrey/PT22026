"""
quiz.py - The "Practice Quiz" learning mode.

The learner is shown a real sentence written by an English learner (from the
JFLEG dataset), tries to correct it, and gets a score by comparing their answer
with the corrections written by 4 human teachers.

We use JFLEG's *validation* split for the quiz and keep the *test* split for
evaluating the model (evaluate.py), so the two never overlap.
"""

import json
import os
import random
import re

from feedback import find_edits, similarity

QUIZ_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "quiz_bank.json")

# Used only if the JFLEG dataset cannot be downloaded (e.g. no internet).
# These example sentences were written for this project.
BUILT_IN_QUESTIONS = [
    {"sentence": "She go to school every day by bus.",
     "corrections": ["She goes to school every day by bus."]},
    {"sentence": "I have visit London two years ago.",
     "corrections": ["I visited London two years ago."]},
    {"sentence": "He is very interesting in learning english.",
     "corrections": ["He is very interested in learning English."]},
    {"sentence": "There is many people in the park today.",
     "corrections": ["There are many people in the park today."]},
    {"sentence": "My brother don't like to eat vegetable.",
     "corrections": ["My brother doesn't like to eat vegetables.",
                     "My brother doesn't like eating vegetables."]},
    {"sentence": "We arrived to the airport at the morning.",
     "corrections": ["We arrived at the airport in the morning."]},
    {"sentence": "I want to buy a umbrella because it rain.",
     "corrections": ["I want to buy an umbrella because it is raining.",
                     "I want to buy an umbrella because it's raining."]},
    {"sentence": "Their house is more bigger than our house.",
     "corrections": ["Their house is bigger than our house.",
                     "Their house is bigger than ours."]},
    {"sentence": "Yesterday we goed to the cinema and watch a film.",
     "corrections": ["Yesterday we went to the cinema and watched a film."]},
    {"sentence": "She can speaks three languages very good.",
     "corrections": ["She can speak three languages very well."]},
    {"sentence": "I am living in this city since 2020.",
     "corrections": ["I have been living in this city since 2020.",
                     "I have lived in this city since 2020."]},
    {"sentence": "Can you give me a advice about my studys?",
     "corrections": ["Can you give me some advice about my studies?",
                     "Can you give me advice about my studies?"]},
]


def detokenize(text: str) -> str:
    """JFLEG writes 'energy .' and 'do n't' - turn that into normal text."""
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _from_jfleg() -> list:
    """Download JFLEG's validation split and keep good quiz sentences."""
    from datasets import load_dataset

    data = load_dataset("jhu-clsp/jfleg", split="validation")
    questions = []
    for row in data:
        sentence = detokenize(row["sentence"])
        corrections = [detokenize(c) for c in row["corrections"] if c.strip()]
        n_words = len(sentence.split())
        has_mistake = all(c.lower() != sentence.lower() for c in corrections)
        if 6 <= n_words <= 25 and has_mistake and corrections:
            questions.append({"sentence": sentence, "corrections": sorted(set(corrections))})
    return questions


def load_questions() -> tuple:
    """Return (questions, source_name). Uses a local cache after the first download."""
    try:
        with open(QUIZ_CACHE, encoding="utf-8") as f:
            return json.load(f), "JFLEG (validation split)"
    except (OSError, json.JSONDecodeError):
        pass
    try:
        questions = _from_jfleg()
        os.makedirs(os.path.dirname(QUIZ_CACHE), exist_ok=True)
        with open(QUIZ_CACHE, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=1)
        return questions, "JFLEG (validation split)"
    except Exception as err:  # no internet, dataset moved, etc.
        print(f"Could not load JFLEG ({err}); using built-in practice sentences.")
        return BUILT_IN_QUESTIONS, "built-in practice sentences"


def pick_question(questions: list, exclude: int = -1) -> int:
    """Pick a random question index (different from the current one)."""
    choices = [i for i in range(len(questions)) if i != exclude] or [0]
    return random.choice(choices)


def grade_answer(question: dict, answer: str) -> dict:
    """Compare the learner's answer with the human corrections.

    Returns a dict with:
      score      0-100, similarity to the closest human correction
      best_ref   the human correction closest to the learner's answer
      remaining  list of Edit objects still needed to reach best_ref
      unchanged  True if the learner did not change the sentence at all
    """
    answer = (answer or "").strip()
    refs = question["corrections"]
    best_ref = max(refs, key=lambda ref: similarity(answer, ref))
    score = round(100 * similarity(answer, best_ref))
    unchanged = similarity(answer, question["sentence"]) == 1.0
    remaining = find_edits(answer, best_ref)
    return {"score": score, "best_ref": best_ref, "remaining": remaining, "unchanged": unchanged}


def feedback_message(result: dict) -> str:
    score = result["score"]
    if result["unchanged"]:
        return "✋ You didn't change the sentence yet - it contains at least one mistake. Try again!"
    if not result["remaining"]:
        return "🏆 **Perfect!** Your answer matches a teacher's correction."
    if score >= 90:
        return f"🌟 **Excellent - {score}/100.** Almost perfect, just a small detail left."
    if score >= 75:
        return f"👍 **Good job - {score}/100.** You fixed most of it."
    if score >= 50:
        return f"🙂 **Nice try - {score}/100.** Some mistakes are still there."
    return f"💪 **Keep practising - {score}/100.** Look at the hints below."
