"""Tests for model architecture parity (T3.2 acceptance criteria).

The checkpoint was trained by a Google Colab notebook whose exported
``MovieMindNCF`` class must be reproduced exactly by
``app.ml.model_architecture.py``. These tests assert the structural contract
(parameter names + shapes + layer layout) so that a mismatch is caught without
running training.
"""

from __future__ import annotations

import torch

from app.ml.model_architecture import MovieMindNCF
from app.ml.model_loader import load_model


def _build() -> MovieMindNCF:
    return MovieMindNCF(
        num_users=610,
        num_movies=9724,
        embedding_dim=32,
        mlp_hidden_dims=(128, 64, 32),
        dropout=0.1,
    )


def test_parameter_names_match_checkpoint():
    """The constructed model and the real checkpoint share the exact keys."""
    model = _build()
    loaded = load_model("artifacts")
    checkpoint_keys = loaded.model.state_dict().keys()
    assert set(model.state_dict().keys()) == set(checkpoint_keys)


def test_embedding_shapes():
    model = _build()
    sd = model.state_dict()
    assert sd["user_gmf.weight"].shape == (610, 32)
    assert sd["movie_gmf.weight"].shape == (9724, 32)
    assert sd["user_mlp.weight"].shape == (610, 32)
    assert sd["movie_mlp.weight"].shape == (9724, 32)


def test_mlp_and_output_shapes():
    model = _build()
    sd = model.state_dict()
    # MLP: Linear(64->128), Linear(128->64), Linear(64->32) at seq indices 0,3,6.
    assert sd["mlp.0.weight"].shape == (128, 64)
    assert sd["mlp.3.weight"].shape == (64, 128)
    assert sd["mlp.6.weight"].shape == (32, 64)
    # Output: Linear(gmf_dim + last_mlp_dim = 32 + 32, 1).
    assert sd["output.weight"].shape == (1, 64)
    assert sd["output.bias"].shape == (1,)


def test_gmf_is_elementwise_product():
    """GMF branch = user_embedding * movie_embedding, producing dim-32 vectors."""
    model = _build()
    u = torch.tensor([0, 1])
    m = torch.tensor([0, 1])
    gmf = model.user_gmf(u) * model.movie_gmf(m)
    assert gmf.shape == (2, 32)


def test_forward_output_is_1d_scores():
    model = _build()
    u = torch.tensor([0, 1, 2])
    m = torch.tensor([5, 6, 7])
    out = model(u, m)
    assert out.shape == (3,)
