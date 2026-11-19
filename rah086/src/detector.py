"""Allergen detector: finds allergen keywords in ingredient text.

Uses exact matching plus fuzzy matching so that small OCR errors
(e.g. 'WHEAL' instead of 'WHEAT') are still detected.
"""

import re
import sys
from difflib import SequenceMatcher

# Allergen groups based on the major food allergens (EU/Codex style lists).
ALLERGENS = {
    "Gluten (cereals)": ["wheat", "gluten", "barley", "rye", "oats", "oat",
                         "spelt", "semolina", "durum", "malt"],
    "Milk": ["milk", "whey", "casein", "caseinate", "lactose", "butter",
             "cream", "cheese", "yoghurt", "yogurt", "ghee"],
    "Eggs": ["egg", "eggs", "albumin", "mayonnaise"],
    "Peanuts": ["peanut", "peanuts", "groundnut", "groundnuts"],
    "Tree nuts": ["almond", "hazelnut", "walnut", "cashew", "pecan",
                  "pistachio", "macadamia", "brazil nut", "tree nuts"],
    "Soy": ["soy", "soya", "soybean", "soybeans", "tofu", "edamame"],
    "Fish": ["fish", "anchovy", "cod", "salmon", "tuna"],
    "Crustaceans": ["crustacean", "crustaceans", "shrimp", "prawn",
                    "crab", "lobster"],
    "Molluscs": ["mollusc", "molluscs", "squid", "oyster", "mussel", "clam"],
    "Sesame": ["sesame", "tahini"],
    "Celery": ["celery", "celeriac"],
    "Mustard": ["mustard"],
    "Lupin": ["lupin", "lupine"],
    "Sulphites": ["sulphite", "sulphites", "sulfite", "sulfites",
                  "sulphur dioxide"],
}

FUZZY_CUTOFF = 0.8  # similarity needed to accept an OCR-damaged word

# Frequent OCR confusions: 'm' is often read as 'rn', and 'w' as 'vv'.
OCR_FIXES = (("rn", "m"), ("vv", "w"))


def tokenize(text):
    """Lowercase the text and split it into alphabetic words."""
    return re.findall(r"[a-z]+", text.lower())


def tokenize_with_ocr_fixes(text):
    """Tokenize the text, plus a copy with common OCR confusions repaired."""
    fixed = text.lower()
    for wrong, right in OCR_FIXES:
        fixed = fixed.replace(wrong, right)
    tokens = tokenize(text)
    for tok in tokenize(fixed):
        if tok not in tokens:
            tokens.append(tok)
    return tokens


def detect_allergens(text, fuzzy_cutoff=FUZZY_CUTOFF):
    """Find allergens mentioned in an ingredient text.

    Args:
        text: ingredient text (for example the OCR output).
        fuzzy_cutoff: minimum similarity (0-1) for fuzzy matches.

    Returns:
        dict mapping allergen name -> sorted list of matched words.
        Fuzzy matches look like 'wheal (~wheat)'.
    """
    if not text or not text.strip():
        return {}

    text_lower = text.lower()
    tokens = tokenize_with_ocr_fixes(text)
    found = {}

    for allergen, keywords in ALLERGENS.items():
        hits = set()
        single_words = {k for k in keywords if " " not in k}

        # Multi-word keywords (e.g. 'tree nuts') are matched as phrases.
        for kw in keywords:
            if " " in kw and kw in text_lower:
                hits.add(kw)

        # Single words: exact match first, then fuzzy match for OCR errors.
        for tok in tokens:
            if tok in single_words:
                hits.add(tok)
                continue
            if len(tok) < 5:
                continue  # short words are too risky for fuzzy matching
            for kw in single_words:
                if (len(kw) >= 5 and tok[0] == kw[0]
                        and SequenceMatcher(None, tok, kw).ratio() >= fuzzy_cutoff):
                    hits.add(f"{tok} (~{kw})")
                    break

        if hits:
            found[allergen] = sorted(hits)
    return found


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.detector <image_path>")
        sys.exit(1)

    from src.ocr import extract_text  # imported here: OCR is slow to load

    ocr_text = extract_text(sys.argv[1])
    print("--- OCR TEXT ---")
    print(ocr_text)
    print("\n--- ALLERGENS FOUND ---")
    results = detect_allergens(ocr_text)
    if not results:
        print("No allergens detected.")
    for name, words in results.items():
        print(f"{name}: {', '.join(words)}")