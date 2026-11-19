"""Automated tests for the validation rules and the prediction pipeline."""

from predictor import load_artifacts, predict_price

SAMPLE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def run_test(name, changes, should_pass):
    """Run one case. Valid inputs must predict; invalid ones must be rejected."""
    data = {**SAMPLE, **changes}

    try:
        prediction = predict_price(data)
    except Exception as error:
        if should_pass:
            print(f"FAIL: {name} | {error}")
            return False
        print(f"PASS: {name} | Rejected correctly")
        return True

    if should_pass:
        print(f"PASS: {name} | ${prediction:,.2f}")
        return True

    print(f"FAIL: {name} | Invalid input accepted")
    return False


TESTS = [
    ("Valid input", {}, True),
    ("Bedrooms exceed rooms", {"total_bedrooms": 1000}, False),
    ("Households exceed population", {"households": 500}, False),
    ("Zero total rooms", {"total_rooms": 0}, False),
    ("Negative median income", {"median_income": -1}, False),
    ("Empty field", {"population": None}, False),
    ("Different ocean category", {"ocean_proximity": "INLAND"}, True),
]

if __name__ == "__main__":
    load_artifacts()
    print("MODEL TEST CASES")
    print("=" * 45)

    results = [run_test(*test) for test in TESTS]

    print("=" * 45)
    print(f"PASSED: {sum(results)}/{len(results)}")
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    raise SystemExit(0 if all(results) else 1)