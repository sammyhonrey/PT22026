"""Risk assessment: combines detected allergens with the user's allergies.

Risk levels:
  HIGH   - the allergen is in the ingredient list
  MEDIUM - the allergen is only in a 'may contain / traces' warning, or is
           mentioned elsewhere on the package (outside the ingredient list)
  LOW    - the allergen was not detected

If an ingredient-list extractor (the fine-tuned NER model) is given, it is
used to tell the ingredient list apart from the rest of the package text.
Without it, every allergen found in the text counts as an ingredient.
"""

import re
import sys

from src.detector import ALLERGENS, detect_allergens

# Text after phrases like "may contain" is treated as a trace warning.
TRACE_PATTERN = re.compile(
    r"(may contain|traces? of|produced in a facility|"
    r"manufactured in a factory)(.*?)(?:storage|best before|\.|$)",
    re.IGNORECASE | re.DOTALL,
)

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def split_text(text):
    """Split label text into (main text, trace warning text)."""
    trace_parts = []

    def grab(match):
        trace_parts.append(match.group(2))
        return " "

    main = TRACE_PATTERN.sub(grab, text)
    return main, " ".join(trace_parts)


def match_user_allergies(user_keywords):
    """Map user keywords (e.g. 'milk') to allergen group names."""
    selected = []
    for kw in user_keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        for name in ALLERGENS:
            if kw in name.lower() and name not in selected:
                selected.append(name)
    return selected


def assess_risk(text, user_allergies, extractor=None):
    """Assess the risk of a product for a user.

    Args:
        text: label text (OCR output or pasted text).
        user_allergies: list of allergen group names the user is allergic to.
        extractor: optional IngredientExtractor (anything with an
            extract(text) method that returns the ingredient list).

    Returns:
        dict with 'overall', 'per_allergen', 'reasons', 'direct', 'traces',
        'elsewhere', 'ingredient_text' and 'used_model'.
    """
    main_text, trace_text = split_text(text)
    anywhere = detect_allergens(main_text)
    traces = detect_allergens(trace_text)

    ingredient_text = extractor.extract(text) if extractor is not None else ""

    if ingredient_text:
        direct = detect_allergens(split_text(ingredient_text)[0])
        elsewhere = {name: words for name, words in anywhere.items()
                     if name not in direct}
        direct_reason = "found in the ingredient list"
    else:
        direct, elsewhere = anywhere, {}
        direct_reason = "found in the label text"

    per_allergen, reasons = {}, {}
    for allergen in user_allergies:
        if allergen in direct:
            per_allergen[allergen] = "HIGH"
            reasons[allergen] = direct_reason
        elif allergen in traces:
            per_allergen[allergen] = "MEDIUM"
            reasons[allergen] = "only in a 'may contain' warning"
        elif allergen in elsewhere:
            per_allergen[allergen] = "MEDIUM"
            reasons[allergen] = ("mentioned outside the ingredient list "
                                 "- check the package")
        else:
            per_allergen[allergen] = "LOW"
            reasons[allergen] = "not detected"

    overall = "LOW"
    for level in per_allergen.values():
        if LEVEL_ORDER[level] > LEVEL_ORDER[overall]:
            overall = level

    return {
        "overall": overall,
        "per_allergen": per_allergen,
        "reasons": reasons,
        "direct": direct,
        "traces": traces,
        "elsewhere": elsewhere,
        "ingredient_text": ingredient_text,
        "used_model": bool(ingredient_text),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python -m src.risk <image_path> "milk,soy"')
        sys.exit(1)

    from src.extractor import IngredientExtractor
    from src.ocr import extract_text

    text = extract_text(sys.argv[1])
    allergies = match_user_allergies(sys.argv[2].split(","))
    print("User allergies:", ", ".join(allergies) or "none matched")

    result = assess_risk(text, allergies, IngredientExtractor())
    print("\nOVERALL RISK:", result["overall"])
    for name, level in result["per_allergen"].items():
        print(f"  {name}: {level} ({result['reasons'][name]})")
    print("\nIngredient list found:", result["ingredient_text"] or "none")
    print("In ingredient list:", ", ".join(result["direct"]) or "none")
    print("Mentioned elsewhere:", ", ".join(result["elsewhere"]) or "none")
    print("Trace warnings:", ", ".join(result["traces"]) or "none")