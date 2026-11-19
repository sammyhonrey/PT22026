"""Evaluate the allergen detector on labelled test cases.

Reports per-case results plus overall precision, recall and F1 score.
Run with:  python -m src.evaluate
"""

from src.detector import detect_allergens

# (ingredient text, set of allergen groups that are truly present)
TEST_CASES = [
    ("Ingredients: wheat flour, sugar, skimmed milk powder, salt.",
     {"Gluten (cereals)", "Milk"}),
    ("Contains soya lecithin, peanuts and sesame seeds.",
     {"Soy", "Peanuts", "Sesame"}),
    ("Water, tomato paste, sugar, salt, spices, vinegar.",
     set()),
    ("Rice flour, palm oil, salt, natural flavouring.",
     set()),
    ("Chocolate: sugar, cocoa butter, whey powder, hazelnuts, emulsifier soy lecithin.",
     {"Milk", "Tree nuts", "Soy"}),
    ("Pasta: durum wheat semolina, egg.",
     {"Gluten (cereals)", "Eggs"}),
    ("Prawn crackers: tapioca starch, prawns, salt, sugar.",
     {"Crustaceans"}),
    ("Dressing: oil, mustard, vinegar, celery seed, sulphites.",
     {"Mustard", "Celery", "Sulphites"}),
    ("Sauce: anchovy, tomato, garlic, olive oil.",
     {"Fish"}),
    ("Dessert: cream, sugar, yoghurt, almond pieces.",
     {"Milk", "Tree nuts"}),
    # Cases with simulated OCR errors
    ("Ingredients: WHEAL flour, sugar, SKIMMED MILK.",
     {"Gluten (cereals)", "Milk"}),
    ("Contains: peanul oil, sesarne seeds.",
     {"Peanuts", "Sesame"}),
    ("Biscuit: wheat flour, rnilk powder, salt.",
     {"Gluten (cereals)", "Milk"}),
    ("Noodles: wheat, soybean oil, hazelnul paste.",
     {"Gluten (cereals)", "Soy", "Tree nuts"}),
]


def evaluate(cases=TEST_CASES, verbose=True):
    tp = fp = fn = 0
    for i, (text, truth) in enumerate(cases, start=1):
        predicted = set(detect_allergens(text).keys())
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        if verbose:
            status = "OK " if predicted == truth else "ERR"
            print(f"[{status}] Case {i}")
            print(f"       expected : {sorted(truth)}")
            print(f"       predicted: {sorted(predicted)}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


if __name__ == "__main__":
    p, r, f = evaluate()
    print(f"\nPrecision: {p:.2f}")
    print(f"Recall   : {r:.2f}")
    print(f"F1 score : {f:.2f}")