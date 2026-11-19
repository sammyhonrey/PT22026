"""Download the CoNLL-2003 test split once and save it locally (data/ is git-ignored)."""
import json
import os

os.environ.pop("HF_HUB_OFFLINE", None)  # this script needs the network

from datasets import load_dataset

try:
    ds = load_dataset("eriktks/conll2003", split="test")
except Exception as err:
    print("Direct load failed, trying the parquet revision:", err)
    ds = load_dataset("eriktks/conll2003", split="test", revision="refs/convert/parquet")

os.makedirs("data", exist_ok=True)
with open("data/conll2003_test.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps({"tokens": row["tokens"], "ner_tags": row["ner_tags"]}) + "\n")
print("Saved", len(ds), "sentences to data/conll2003_test.jsonl")