"""Fine-tune the SST-2 DistilBERT model on 2,000 IMDB training reviews.

Usage:
    python train.py

Saves the model to models/finetuned-imdb (not uploaded to GitHub) and the
training details to eval_results/training_info.json.
"""
import json
import os
import time

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
OUT_MODEL_DIR = "models/finetuned-imdb"
N_TRAIN = 2000
MAX_LEN = 256
SEED = 42

os.makedirs("eval_results", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

train_data = (
    load_dataset("stanfordnlp/imdb", split="train")
    .shuffle(seed=SEED)
    .select(range(N_TRAIN))
)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)


train_data = train_data.map(tokenize, batched=True)
train_data = train_data.remove_columns(["text"])

args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    data_collator=DataCollatorWithPadding(tokenizer),
    processing_class=tokenizer,
)

start = time.time()
result = trainer.train()
minutes = (time.time() - start) / 60

trainer.save_model(OUT_MODEL_DIR)
tokenizer.save_pretrained(OUT_MODEL_DIR)

with open("eval_results/training_info.json", "w") as f:
    json.dump(
        {
            "base_model": BASE_MODEL,
            "train_samples": N_TRAIN,
            "max_length": MAX_LEN,
            "epochs": 1,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "final_train_loss": result.training_loss,
            "training_minutes": round(minutes, 1),
        },
        f,
        indent=2,
    )

print(f"Done in {minutes:.1f} minutes. Model saved to {OUT_MODEL_DIR}")