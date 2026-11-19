"""Unit tests for the allergen detector and risk assessment.

Run from the rah086 folder with:  python -m unittest discover tests
"""

import unittest

from src.detector import detect_allergens
from src.risk import assess_risk, match_user_allergies, split_text


class TestDetector(unittest.TestCase):
    def test_exact_match(self):
        result = detect_allergens("Contains milk and peanuts.")
        self.assertIn("Milk", result)
        self.assertIn("Peanuts", result)

    def test_no_allergens(self):
        self.assertEqual(detect_allergens("Water, salt, sugar."), {})

    def test_empty_text(self):
        self.assertEqual(detect_allergens(""), {})
        self.assertEqual(detect_allergens("   "), {})

    def test_case_insensitive(self):
        self.assertIn("Soy", detect_allergens("SOYBEAN OIL"))

    def test_fuzzy_match_ocr_error(self):
        # 'WHEAL' is a typical OCR misreading of 'WHEAT'.
        self.assertIn("Gluten (cereals)", detect_allergens("WHEAL flour"))

    def test_rn_confusion_fixed(self):
        # 'sesarne' is 'sesame' with m read as rn.
        self.assertIn("Sesame", detect_allergens("sesarne seeds"))

    def test_multi_word_keyword(self):
        self.assertIn("Tree nuts", detect_allergens("may contain tree nuts"))


class TestRisk(unittest.TestCase):
    TEXT = "Ingredients: wheat flour, soy. May contain fish and egg."

    def test_split_text(self):
        main, trace = split_text(self.TEXT)
        self.assertIn("soy", main)
        self.assertIn("fish", trace)

    def test_direct_ingredient_is_high(self):
        result = assess_risk(self.TEXT, ["Soy"])
        self.assertEqual(result["per_allergen"]["Soy"], "HIGH")

    def test_trace_is_medium(self):
        result = assess_risk(self.TEXT, ["Fish"])
        self.assertEqual(result["per_allergen"]["Fish"], "MEDIUM")

    def test_absent_is_low(self):
        result = assess_risk(self.TEXT, ["Sesame"])
        self.assertEqual(result["per_allergen"]["Sesame"], "LOW")
        self.assertEqual(result["overall"], "LOW")

    def test_overall_is_highest(self):
        result = assess_risk(self.TEXT, ["Soy", "Fish", "Sesame"])
        self.assertEqual(result["overall"], "HIGH")

    def test_match_user_allergies(self):
        self.assertEqual(match_user_allergies(["milk", " soy "]),
                         ["Milk", "Soy"])


if __name__ == "__main__":
    unittest.main()