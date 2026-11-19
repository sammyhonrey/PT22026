"""Export the first trees of the trained Random Forest to JSON for the browser demo."""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features import FEATURE_NAMES, extract_features  

HF_REPO = "Ratul16961/phishing-url-random-forest"
HF_FILE = "best_model.joblib"
N_TREES = int(os.environ.get("N_TREES", "20"))
OUT_DIR = ROOT / "web_demo"

CHECK_URLS = [
    "https://github.com",
    "www.wikipedia.org",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.ynu.edu.cn",
    "https://www.google.com/search?q=python",
    "http://paypa1-secure-login.xyz/verify@account",
    "paypa1-secure-login.xyz",
    "http://secure-paypal-verify-account.tk/login.php?id=123456",
    "http://192.168.1.15/bank/login.php",
    "http://user@evil.example.com:8080/a/b;c=1?x=1&y=2#top",
    "mxp2216.com",
    "http://paypal.com.account-update.example.net//login/verify.php",
]


def load_model():
    for name in ("models/best_model.joblib", "best_model.joblib"):
        path = ROOT / name
        if path.exists():
            return joblib.load(path)
    from huggingface_hub import hf_hub_download

    return joblib.load(hf_hub_download(HF_REPO, HF_FILE))


def export_tree(estimator):
    """Compact form of one tree. The left child of node i is always node i+1."""
    t = estimator.tree_
    left, right = t.children_left, t.children_right
    internal = left != -1
    idx = np.arange(t.node_count)
    assert np.all(left[internal] == idx[internal] + 1), "unexpected node order in the tree"
    values = t.value[:, 0, :]
    p_phishing = values[:, 1] / values.sum(axis=1)
    return {
        "f": np.where(internal, t.feature, -1).astype(int).tolist(),
        "t": [round(float(x), 4) if keep else 0 for x, keep in zip(t.threshold, internal)],
        "r": np.where(internal, right, 0).astype(int).tolist(),
        "p": np.rint(p_phishing * 1000).astype(int).tolist(),
    }


def main():
    model = load_model()
    n = min(N_TREES, len(model.estimators_))
    trees = [export_tree(e) for e in model.estimators_[:n]]

    X = pd.DataFrame([extract_features(u) for u in CHECK_URLS], columns=FEATURE_NAMES)
    full = model.predict_proba(X)[:, 1]
    X32 = X.to_numpy(dtype=np.float32)
    subset = np.mean([e.predict_proba(X32)[:, 1] for e in model.estimators_[:n]], axis=0)

    payload = {
        "n_trees": n,
        "feature_names": FEATURE_NAMES,
        "trees": trees,
        "check": [
            {"url": u, "subset": round(float(s), 4), "full": round(float(f), 4)}
            for u, s, f in zip(CHECK_URLS, subset, full)
        ],
    }
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "model.json"
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    print(f"Exported {n} of {len(model.estimators_)} trees to {out}")
    print(f"File size: {out.stat().st_size / 1e6:.1f} MB")
    print(f"\n{'full (100 trees)':>17} {f'subset ({n} trees)':>17}  URL")
    for u, s, f in zip(CHECK_URLS, subset, full):
        print(f"{f:17.3f} {s:17.3f}  {u}")
    if out.stat().st_size > 10e6:
        print("\nThe file is larger than 10 MB. Run again with a smaller number, for example:")
        print("  N_TREES=10 python tools/export_web_model.py")


if __name__ == "__main__":
    main()