from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

text = "Elon Musk visited Google headquarters in California last Monday."
for ent in ner(text):
    print(ent["entity_group"], "|", ent["word"], "|", round(float(ent["score"]), 3))