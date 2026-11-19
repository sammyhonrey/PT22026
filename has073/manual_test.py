import joblib
import pandas as pd

from features import FEATURE_NAMES, extract_features

model = joblib.load("models/best_model.joblib")

TESTS = [
    ("https://github.com", 0),
    ("www.wikipedia.org", 0),
    ("https://en.wikipedia.org/wiki/Machine_learning", 0),
    ("https://www.ynu.edu.cn", 0),
    ("https://www.google.com/search?q=python", 0),
    ("http://paypa1-secure-login.xyz/verify@account", 1),
    ("paypa1-secure-login.xyz", 1),
    ("http://secure-paypal-verify-account.tk/login.php?id=123456", 1),
    ("http://192.168.1.15/bank/login.php", 1),
]

correct = 0
for url, truth in TESTS:
    feats = pd.DataFrame([extract_features(url)], columns=FEATURE_NAMES)
    prob = model.predict_proba(feats)[0, 1]
    pred = int(prob >= 0.5)
    correct += pred == truth
    status = "OK " if pred == truth else "BAD"
    expected = "Phishing" if truth else "Safe"
    print(f"{status} phishing prob {prob:.2f}  expected {expected:8s}  {url}")

print(f"\n{correct}/{len(TESTS)} correct")