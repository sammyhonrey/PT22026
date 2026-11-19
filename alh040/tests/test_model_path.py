"""
Unit tests for how GrammarCoach chooses where to load the T5 model from.

The model folder `models/t5-grammar` is not stored in Git (it is ~890 MB).
grammar_model.py must therefore use the local folder when it exists and
fall back to the Hugging Face Hub when it does not.

Run:   python -m pytest tests/
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grammar_model  # noqa: E402

HUB_ID = "vennify/t5-base-grammar-correction"


def _reload_with_local_folder(monkeypatch, exists):
    """Reload grammar_model while pretending the local model folder exists or not."""
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        os.path, "isdir",
        lambda p: exists if p.endswith(os.path.join("models", "t5-grammar")) else real_isdir(p),
    )
    return importlib.reload(grammar_model)


def test_local_model_path_is_inside_project_folder():
    project_dir = os.path.dirname(os.path.abspath(grammar_model.__file__))
    assert grammar_model._LOCAL_MODEL == os.path.join(project_dir, "models", "t5-grammar")
    assert os.path.isabs(grammar_model._LOCAL_MODEL)


def test_uses_local_folder_when_it_exists(monkeypatch):
    gm = _reload_with_local_folder(monkeypatch, exists=True)
    assert gm.MODEL_NAME == gm._LOCAL_MODEL


def test_falls_back_to_hub_when_local_folder_is_missing(monkeypatch):
    gm = _reload_with_local_folder(monkeypatch, exists=False)
    assert gm.MODEL_NAME == HUB_ID


def teardown_module(module):
    # restore the real module state for any tests that run afterwards
    importlib.reload(grammar_model)
