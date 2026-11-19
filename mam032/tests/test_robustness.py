"""Expanding-origin checks exclude every reserved future target before fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, build_supervised_dataset
from src.robustness import development_data, expanding_folds, run_development_diagnostics

PROTOCOL = json.loads((Path(__file__).parents[1] / "config/robustness_protocol.json").read_text())
METADATA = {"boundaries": {"validation_end": "2023-02-09 23:00", "test_start": "2023-02-10 23:00"}}


def test_future_mutation_cannot_change_development_results(hourly_series):
    fitted = []

    def factory():
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10))
        fitted.append(estimator)
        return {"Ridge": estimator}

    first = run_development_diagnostics(hourly_series, METADATA, PROTOCOL, factory)
    changed = hourly_series.copy()
    changed.loc[changed.index > pd.Timestamp(METADATA["boundaries"]["test_start"]), "energy_kwh"] = 1e9
    second = run_development_diagnostics(changed, METADATA, PROTOCOL, factory)
    assert first["development_fingerprint"] == second["development_fingerprint"]
    assert_frame_equal(first["predictions"], second["predictions"])
    assert_frame_equal(first["fold_metrics"], second["fold_metrics"])
    assert len({id(model) for model in fitted}) == 6
    development, _ = development_data(hourly_series, METADATA)
    rows = build_supervised_dataset(development)
    evaluated = set()
    for fold, model in zip(first["folds"], fitted[:3]):
        origins = pd.to_datetime(fold["train_origins"])
        train = rows.loc[rows.forecast_origin.isin(origins)]
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, train[FEATURE_COLUMNS].mean())
        assert fold["training_threshold_kwh"] == train.target_kwh.quantile(.9)
        assert pd.Timestamp(fold["last_training_target"]) < pd.Timestamp(fold["first_evaluation_target"])
        current = set(fold["evaluation_origins"])
        assert not (current & evaluated)
        evaluated |= current
    assert first["predictions"].target_timestamp.max() <= pd.Timestamp(first["development_cutoff"])
    assert first["predictions"].groupby(["model", "fold", "forecast_origin"]).size().eq(24).all()


def test_overlapping_reserved_test_boundary_is_rejected(hourly_series):
    metadata = {"boundaries": {"validation_end": "2023-02-09 23:00", "test_start": "2023-02-10 00:00"}}
    with pytest.raises(ValueError, match="overlaps"):
        development_data(hourly_series, metadata)


def test_purge_removes_origins_with_unavailable_training_targets(hourly_series):
    data = build_supervised_dataset(hourly_series)
    first_evaluation = expanding_folds(data)[0]["evaluation_origins"][0]
    last_training = data.loc[data.forecast_origin < first_evaluation, "forecast_origin"].max()
    data.loc[data.forecast_origin.eq(last_training), "target_timestamp"] = first_evaluation + pd.Timedelta(hours=1)
    assert last_training not in expanding_folds(data)[0]["train_origins"]


@pytest.mark.parametrize("count,fraction", [(True, .5), (1, .5), (3, 0), (3, 1), (300, .5)])
def test_invalid_fold_protocol_cannot_produce_a_result(hourly_series, count, fraction):
    with pytest.raises(ValueError):
        expanding_folds(build_supervised_dataset(hourly_series), count, fraction)
