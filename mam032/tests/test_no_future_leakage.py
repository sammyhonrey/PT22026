import pandas as pd
from pandas.testing import assert_frame_equal

from src.features import FEATURE_COLUMNS, features_for_origin


def test_future_changes_do_not_change_origin_features(hourly_series: pd.DataFrame) -> None:
    origin = hourly_series.index[24 * 20 + 23]
    original = features_for_origin(hourly_series, origin)[FEATURE_COLUMNS]
    changed = hourly_series.copy()
    changed.loc[changed.index > origin, "energy_kwh"] = 9999.0
    mutated = features_for_origin(changed, origin)[FEATURE_COLUMNS]
    assert_frame_equal(original, mutated)


def test_target_aligned_daily_lag_never_exceeds_origin(hourly_series: pd.DataFrame) -> None:
    origin = hourly_series.index[24 * 20 + 23]
    features = features_for_origin(hourly_series, origin)
    assert (features["target_timestamp"] - pd.Timedelta(hours=24) <= origin).all()
