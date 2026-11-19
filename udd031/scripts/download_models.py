#!/usr/bin/env python
"""Pre-download and verify the Hugging Face weights.

Moves the first-run download to a setup step and checks that each checkpoint still
advertises the label vocabulary the ontology was written against.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mmsa.config import load_config  # noqa: E402

# Verified against the Hub on 2026-11-04. If a check fails, the model changed --
# do not silently proceed, because the fusion maths assumes these vocabularies.
EXPECTED_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "acoustic": {
        "superb/wav2vec2-base-superb-er": ("ang", "hap", "neu", "sad"),
    },
    "text": {
        "bhadresh-savani/distilbert-base-uncased-emotion": (
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise",
        ),
    },
}


def _plan(cfg) -> list[tuple[str, str]]:
    """Return ``[(role, model_id)]`` for everything the app can load."""
    rows = [
        ("asr", cfg.models.asr_id),
        ("acoustic", cfg.models.acoustic_id),
        ("text", cfg.models.text_id),
    ]
    if cfg.stress.dimensional_enabled:
        rows.append(("dimensional", cfg.models.dimensional_id))
    return rows


def _download(role: str, model_id: str, cache_dir: str | None) -> bool:
    """Fetch one checkpoint into the local Hub cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub is not installed. Run: pip install -r requirements.txt")
        return False

    print(f"  {role:<12} {model_id}")
    try:
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.bin",
                "*.safetensors",
                "*.yaml",
                "*.yml",
            ],
        )
    except Exception as exc:  # noqa: BLE001 - network and auth failures alike
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    print(f"    cached at {path}")
    return True


def _verify(cfg) -> int:
    """Compare each classifier's id2label against the expected vocabulary."""
    try:
        from transformers import AutoConfig
    except ImportError:
        print("transformers is not installed; cannot verify. Run: pip install -r requirements.txt")
        return 2

    failures = 0
    for role, table in EXPECTED_LABELS.items():
        model_id = cfg.models.acoustic_id if role == "acoustic" else cfg.models.text_id
        expected = table.get(model_id)
        if expected is None:
            print(f"  {role:<12} no expectation recorded for {model_id}; skipped")
            continue
        try:
            hf_config = AutoConfig.from_pretrained(model_id, cache_dir=cfg.cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  {role:<12} could not read config: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        id2label = getattr(hf_config, "id2label", None) or {}
        found = tuple(str(id2label.get(i, f"LABEL_{i}")).lower() for i in range(len(id2label)))
        normalised = tuple(label.strip().lower() for label in expected)
        ok = found == normalised
        print(f"  {role:<12} {model_id}")
        print(f"    expected {list(normalised)}")
        print(f"    found    {list(found)}  ->  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures += 1
            print(
                "    Fix config/ontology.yaml before running the app: the adapter maps\n"
                "    these exact strings, and an unrecognised label would be dropped as\n"
                "    ignorance rather than raising, which would look like a bug."
            )

    print()
    print("Verification passed." if not failures else f"{failures} verification failure(s).")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without downloading")
    parser.add_argument("--verify", action="store_true", help="check label vocabularies only")
    parser.add_argument("--only", choices=["asr", "acoustic", "text", "dimensional"], help="fetch one model")
    args = parser.parse_args(argv)

    cfg = load_config()
    cache_dir = os.environ.get("MMSA_CACHE_DIR") or cfg.cache_dir

    print("Multimodal Mental State Analyser - model acquisition")
    print(f"device     : {cfg.device}")
    print(f"cache dir  : {cache_dir or 'Hugging Face default (~/.cache/huggingface)'}")
    print(f"dimensional: {'enabled' if cfg.stress.dimensional_enabled else 'disabled'}")
    print()

    if args.verify:
        return _verify(cfg)

    rows = [(r, m) for r, m in _plan(cfg) if not args.only or r == args.only]
    if args.dry_run:
        print("Plan:")
        for role, model_id in rows:
            print(f"  {role:<12} {model_id}")
        print("\nDry run: nothing downloaded.")
        return 0

    failures = 0
    for role, model_id in rows:
        if not _download(role, model_id, cache_dir):
            failures += 1

    print()
    if failures:
        print(f"{failures} model(s) failed to download.")
        return 1
    print("All requested models are cached. Running label verification next.\n")
    return _verify(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
