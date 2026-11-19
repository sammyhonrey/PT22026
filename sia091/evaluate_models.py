import csv
import gc
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from src.chunker import chunk_pages
from src.embedder import Embedder
from src.loader import load_document
from src.retriever import Retriever


DEFAULT_DOCUMENT = "data/assignment_guidelines.pdf"
EVAL_SET = "eval/eval_set.json"
RESULTS_FILE = "eval/controlled_model_comparison.csv"

MODELS = {
    "DistilBERT": "distilbert/distilbert-base-cased-distilled-squad",
    "RoBERTa": "deepset/roberta-base-squad2",
}

CHUNK_SIZES = (40, 60, 120)
OVERLAP = 10
TOP_KS = (1, 3, 5)
MAX_ANSWER_TOKENS = 15
MAX_SEQUENCE_LENGTH = 384
RETRIEVAL_WEIGHT = 5.0

STRATEGIES = (
    "retrieval_first",
    "qa_prob",
    "qa_logit",
    "retrieval_x_prob",
    "logit_plus_retrieval",
)


def normalize_text(text):
    text = str(text).lower()
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(
        normalize_text(prediction) == normalize_text(reference)
    )


def token_f1(prediction, reference):
    predicted_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    common = Counter(predicted_tokens) & Counter(reference_tokens)
    common_count = sum(common.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(predicted_tokens)
    recall = common_count / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def permissive_correct(prediction, reference):
    predicted = normalize_text(prediction)
    expected = normalize_text(reference)

    if not predicted or not expected:
        return 0.0

    if expected in predicted:
        return 1.0

    predicted_tokens = predicted.split()
    expected_tokens = set(expected.split())

    return float(
        all(token in expected_tokens for token in predicted_tokens)
    )


def evidence_found(evidence, chunks):
    if not evidence:
        return 0.0

    if isinstance(evidence, str):
        evidence_items = [evidence]
    else:
        evidence_items = evidence

    retrieved_text = " ".join(
        chunk["text"] for chunk in chunks
    )

    retrieved_text = " ".join(retrieved_text.lower().split())

    for item in evidence_items:
        expected = " ".join(str(item).lower().split())

        if expected and expected in retrieved_text:
            return 1.0

    return 0.0


class ControlledQA:

    def __init__(self, model_name, device):
        self.device = device
        self.max_answer_tokens = MAX_ANSWER_TOKENS

        print("Loading tokenizer:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading QA model:", model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name
        )

        self.model.to(device)
        self.model.eval()

        print("Model loaded on:", device)

    def answer_one(self, question, context):

        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=MAX_SEQUENCE_LENGTH,
            return_offsets_mapping=True,
        )

        offsets = encoded.pop("offset_mapping")[0]
        sequence_ids = encoded.sequence_ids(0)

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = self.model(**model_inputs)

        in_context = torch.tensor(
            [sid == 1 for sid in sequence_ids],
            device=self.device,
            dtype=torch.bool,
        )

        start_logits = outputs.start_logits[0].masked_fill(
            ~in_context, -1e9
        )

        end_logits = outputs.end_logits[0].masked_fill(
            ~in_context, -1e9
        )

        size = start_logits.shape[0]

        band = torch.triu(
            torch.tril(
                torch.ones(
                    size,
                    size,
                    device=self.device,
                ),
                diagonal=self.max_answer_tokens - 1,
            )
        ).bool()

        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        prob_spans = (
            start_probs[:, None] * end_probs[None, :]
        ).masked_fill(~band, 0.0)

        logit_spans = (
            start_logits[:, None] + end_logits[None, :]
        ).masked_fill(~band, -1e18)

        best_index = int(logit_spans.argmax().item())

        start_token = best_index // size
        end_token = best_index % size

        start_char = int(offsets[start_token][0])
        end_char = int(offsets[end_token][1])

        answer = context[start_char:end_char].strip()

        return {
            "answer": answer,
            "prob": float(
                prob_spans[start_token, end_token].item()
            ),
            "logit": float(
                logit_spans[start_token, end_token].item()
            ),
        }

    def candidates(self, question, chunks):

        results = []

        for chunk in chunks:
            found = self.answer_one(
                question,
                chunk["text"],
            )

            found["page"] = chunk["page"]
            found["retrieval"] = chunk.get("score", 0.0)

            results.append(found)

        return results


def select_candidate(candidates, strategy):

    if not candidates:
        return {"answer": ""}

    if strategy == "retrieval_first":
        return candidates[0]

    if strategy == "qa_prob":
        return max(candidates, key=lambda item: item["prob"])

    if strategy == "qa_logit":
        return max(candidates, key=lambda item: item["logit"])

    if strategy == "retrieval_x_prob":
        return max(
            candidates,
            key=lambda item: item["retrieval"] * item["prob"],
        )

    if strategy == "logit_plus_retrieval":
        return max(
            candidates,
            key=lambda item: (
                item["logit"]
                + RETRIEVAL_WEIGHT * item["retrieval"]
            ),
        )

    raise ValueError("Unknown strategy: " + strategy)


def main():

    document_path = (
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCUMENT
    )

    if not Path(document_path).exists():
        raise FileNotFoundError(
            "Document not found: " + document_path
        )

    if not Path(EVAL_SET).exists():
        raise FileNotFoundError(
            "Evaluation file not found: " + EVAL_SET
        )

    with open(EVAL_SET, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    print("Loading document...")
    pages = load_document(document_path)

    print("Loading embedding model...")
    embedder = Embedder()

    experiment_sets = {}

    print("Preparing retrieval results...")

    for chunk_size in CHUNK_SIZES:

        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=OVERLAP,
        )

        retriever = Retriever(embedder)
        retriever.build(chunks)

        for top_k in TOP_KS:

            config_results = []

            for item in questions:
                question = item["question"]

                retrieved = retriever.search(
                    question,
                    top_k=top_k,
                )

                config_results.append({
                    "question": question,
                    "reference": item["answer"],
                    "evidence": item.get("evidence", ""),
                    "chunks": retrieved,
                    "recall": evidence_found(
                        item.get("evidence", ""),
                        retrieved,
                    ),
                })

            experiment_sets[(chunk_size, top_k)] = config_results

        del retriever
        gc.collect()

    output_path = Path(RESULTS_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "model_id",
        "chunk_size",
        "overlap",
        "top_k",
        "strategy",
        "questions",
        "recall_at_k",
        "permissive_accuracy",
        "exact_match",
        "token_f1",
    ]

    print("Results will be saved to:", output_path)

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        output_file.flush()

        for model_label, model_name in MODELS.items():

            print()
            print("Testing model:", model_label)

            qa = ControlledQA(model_name, device)

            for chunk_size in CHUNK_SIZES:
                for top_k in TOP_KS:

                    config_results = experiment_sets[
                        (chunk_size, top_k)
                    ]

                    metric_values = {
                        strategy: {
                            "permissive": [],
                            "exact": [],
                            "f1": [],
                        }
                        for strategy in STRATEGIES
                    }

                    recalls = []

                    for item in config_results:

                        question = item["question"]
                        reference = item["reference"]
                        chunks = item["chunks"]

                        recalls.append(item["recall"])

                        candidates = qa.candidates(
                            question,
                            chunks,
                        )

                        for strategy in STRATEGIES:

                            selected = select_candidate(
                                candidates,
                                strategy,
                            )

                            prediction = selected["answer"]

                            metric_values[strategy][
                                "permissive"
                            ].append(
                                permissive_correct(
                                    prediction,
                                    reference,
                                )
                            )

                            metric_values[strategy][
                                "exact"
                            ].append(
                                exact_match(
                                    prediction,
                                    reference,
                                )
                            )

                            metric_values[strategy][
                                "f1"
                            ].append(
                                token_f1(
                                    prediction,
                                    reference,
                                )
                            )

                    recall_score = sum(recalls) / len(recalls)

                    for strategy in STRATEGIES:

                        values = metric_values[strategy]
                        count = len(config_results)

                        row = {
                            "model": model_label,
                            "model_id": model_name,
                            "chunk_size": chunk_size,
                            "overlap": OVERLAP,
                            "top_k": top_k,
                            "strategy": strategy,
                            "questions": count,
                            "recall_at_k": round(
                                recall_score, 4
                            ),
                            "permissive_accuracy": round(
                                sum(values["permissive"]) / count,
                                4,
                            ),
                            "exact_match": round(
                                sum(values["exact"]) / count,
                                4,
                            ),
                            "token_f1": round(
                                sum(values["f1"]) / count,
                                4,
                            ),
                        }

                        writer.writerow(row)

                    output_file.flush()

                    print(
                        model_label,
                        "| chunk:", chunk_size,
                        "| top_k:", top_k,
                        "| Recall@K:", round(recall_score, 4),
                    )

            del qa
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(model_label, "completed.")

    print()
    print("Controlled experiment completed.")
    print("Results:", output_path)


if __name__ == "__main__":
    main()
