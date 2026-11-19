"""Fine-tune a multilingual transformer to find the ingredient list in OCR text.

Each word of a label text gets one tag:
    0 = outside, 1 = start of the ingredient list, 2 = inside the ingredient list.

Quick test run : python -m src.train_ner --max-train 200 --max-test 100 --epochs 1
Full training  : python -m src.train_ner --epochs 3
"""

import argparse
import gzip
import json
import random
import time
from pathlib import Path

MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_DIR = Path("data/raw")
OUT_DIR = Path("models/ingredient-ner")
METRICS_PATH = Path("docs/ner_metrics.json")
CHUNK_WORDS = 100   # long texts are split into chunks of this many words
MAX_LEN = 256       # maximum number of sub-word tokens per chunk
LABELS = ["O", "B-ING", "I-ING"]


def load_jsonl_gz(path, limit=None, seed=42):
    """Read a .jsonl.gz file. If limit is given, return a random sample."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit and limit < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def make_chunks(rows, chunk_words=CHUNK_WORDS):
    """Split every document into chunks of at most chunk_words words."""
    chunks = []
    for row in rows:
        words, tags = row["tokens"], row["ner_tags"]
        for i in range(0, len(words), chunk_words):
            chunks.append((words[i:i + chunk_words], tags[i:i + chunk_words]))
    return chunks


def align_labels(word_ids, tags):
    """Give the word's tag to its first sub-token; ignore the rest (-100)."""
    labels, previous = [], None
    for wid in word_ids:
        if wid is None or wid == previous:
            labels.append(-100)
        else:
            labels.append(tags[wid])
        previous = wid
    return labels


def encode(chunks, tokenizer, max_len=MAX_LEN):
    """Turn (words, tags) chunks into model inputs."""
    items = []
    for words, tags in chunks:
        enc = tokenizer(words, is_split_into_words=True,
                        truncation=True, max_length=max_len)
        items.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": align_labels(enc.word_ids(), tags),
            "n_words": len(words),
        })
    return items


def binary_metrics(gold, pred):
    """Precision / recall / F1 for 'word is part of the ingredient list'."""
    tp = sum(1 for g, p in zip(gold, pred) if g > 0 and p > 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p > 0)
    fn = sum(1 for g, p in zip(gold, pred) if g > 0 and p == 0)
    correct = sum(1 for g, p in zip(gold, pred) if (g > 0) == (p > 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(correct / len(gold), 4) if gold else 0.0,
        "words_evaluated": len(gold),
    }


def make_collate(pad_id):
    import torch

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            mask[i, :n] = torch.tensor(b["attention_mask"])
            labels[i, :n] = torch.tensor(b["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


def evaluate(model, items, collate, device, batch_size=32):
    """Word-level evaluation on encoded items."""
    import torch

    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            guesses = logits.argmax(dim=-1)
            keep = batch["labels"] != -100
            gold.extend(batch["labels"][keep].tolist())
            pred.extend(guesses[keep].tolist())
    return binary_metrics(gold, pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train", type=int, default=None,
                        help="use only N random training documents")
    parser.add_argument("--max-test", type=int, default=None,
                        help="use only N random test documents")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_rows = load_jsonl_gz(
        DATA_DIR / "ingredient_detection_dataset_train.jsonl.gz", args.max_train)
    test_rows = load_jsonl_gz(
        DATA_DIR / "ingredient_detection_dataset_test.jsonl.gz", args.max_test)
    print(f"Documents: train={len(train_rows)} test={len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_items = encode(make_chunks(train_rows), tokenizer)
    test_items = encode(make_chunks(test_rows), tokenizer)
    print(f"Chunks: train={len(train_items)} test={len(test_items)}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)})
    model.to(device)

    collate = make_collate(tokenizer.pad_token_id)
    steps_per_epoch = (len(train_items) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    start_time, step = time.time(), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_items)
        for start in range(0, len(train_items), args.batch_size):
            batch = collate(train_items[start:start + args.batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0 or step == total_steps:
                elapsed = time.time() - start_time
                eta = elapsed / step * (total_steps - step)
                print(f"epoch {epoch}/{args.epochs} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} "
                      f"elapsed {elapsed/60:.1f} min, about {eta/60:.1f} min left")

    metrics = evaluate(model, test_items, collate, device)
    metrics.update({
        "model": args.model,
        "epochs": args.epochs,
        "train_documents": len(train_rows),
        "test_documents": len(test_rows),
        "training_minutes": round((time.time() - start_time) / 60, 1),
    })
    print("\nTEST RESULTS (is each word part of the ingredient list?)")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nModel saved to {args.out}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()