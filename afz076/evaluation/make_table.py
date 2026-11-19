"""Print the main robustness results as a Markdown table (paste it into the README or the report).

Usage:  python evaluation/make_table.py            (reads evaluation/results/robustness.csv)
Shows accuracy in % on the official test set: clean, then each weather condition at severity 5.
Glare and severity 5 were never seen in training.
"""
import sys
from pathlib import Path

import pandas as pd

csv = Path("evaluation/results/robustness.csv")
if not csv.exists():
    sys.exit("Run evaluation/evaluate.py first (it writes evaluation/results/robustness.csv).")

df = pd.read_csv(csv)
clean = df[df.condition == "clean"].set_index("run")["accuracy"]
sev5 = df[df.severity == 5].pivot_table(index="run", columns="condition", values="accuracy")
table = (pd.concat([clean.rename("clean"), sev5], axis=1) * 100).round(1)
cols = ["clean", "fog", "night", "rain", "blur", "noise", "glare"]
table = table[[c for c in cols if c in table.columns]]

print("| Run | " + " | ".join(table.columns) + " |")
print("|---|" + "---|" * len(table.columns))
for run, row in table.iterrows():
    print(f"| {run} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
