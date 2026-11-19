# Results on 3453 random CoNLL-2003 test sentences (seed 42)

| Model | Precision | Recall | F1 | Time (s) |
|---|---|---|---|---|
| BERT pre-trained (dslim/bert-base-NER) | 0.864 | 0.915 | 0.888 | 215 |
| BERT fine-tuned by us | 0.891 | 0.912 | 0.901 | 211 |
| spaCy en_core_web_sm (mapped labels) | 0.656 | 0.545 | 0.595 | 27 |

## Per-entity F1

| Model | PER | ORG | LOC | MISC |
|---|---|---|---|---|
| BERT pre-trained (dslim/bert-base-NER) | 0.887 | 0.891 | 0.926 | 0.799 |
| BERT fine-tuned by us | 0.951 | 0.890 | 0.920 | 0.781 |
| spaCy en_core_web_sm (mapped labels) | 0.646 | 0.360 | 0.749 | 0.614 |
