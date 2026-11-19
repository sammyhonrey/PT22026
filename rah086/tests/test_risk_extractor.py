"""Tests for risk assessment with an ingredient-list extractor.

A fake extractor is used so the tests run without the trained model.
Run from the rah086 folder with:  python -m unittest discover tests
"""

import unittest

from src.risk import assess_risk


class FakeExtractor:
    """Pretends to be the NER model and returns a fixed ingredient list."""

    def __init__(self, ingredient_list):
        self.ingredient_list = ingredient_list

    def extract(self, text):
        return self.ingredient_list


TEXT = ("Gluten free snack! Ingredients: rice, milk powder. "
        "May contain peanuts. Storage: keep cool.")


class TestRiskWithExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FakeExtractor("rice, milk powder")

    def test_allergen_in_ingredient_list_is_high(self):
        result = assess_risk(TEXT, ["Milk"], self.extractor)
        self.assertEqual(result["per_allergen"]["Milk"], "HIGH")
        self.assertTrue(result["used_model"])

    def test_mention_outside_list_is_medium(self):
        # 'Gluten free' is on the pack but not in the ingredient list.
        result = assess_risk(TEXT, ["Gluten (cereals)"], self.extractor)
        self.assertEqual(result["per_allergen"]["Gluten (cereals)"], "MEDIUM")

    def test_trace_warning_is_medium(self):
        result = assess_risk(TEXT, ["Peanuts"], self.extractor)
        self.assertEqual(result["per_allergen"]["Peanuts"], "MEDIUM")

    def test_absent_allergen_is_low(self):
        result = assess_risk(TEXT, ["Soy"], self.extractor)
        self.assertEqual(result["per_allergen"]["Soy"], "LOW")

    def test_empty_extraction_falls_back_to_full_text(self):
        result = assess_risk(TEXT, ["Milk"], FakeExtractor(""))
        self.assertEqual(result["per_allergen"]["Milk"], "HIGH")
        self.assertFalse(result["used_model"])

    def test_works_without_extractor(self):
        result = assess_risk(TEXT, ["Milk"])
        self.assertEqual(result["per_allergen"]["Milk"], "HIGH")
        self.assertFalse(result["used_model"])


if __name__ == "__main__":
    unittest.main()