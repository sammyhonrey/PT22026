import pandas as pd

from src.evaluation import chronological_origin_split
from src.features import build_supervised_dataset


def test_split_is_strictly_chronological(hourly_series: pd.DataFrame) -> None:
    dataset = build_supervised_dataset(hourly_series)
    train, validation, test = chronological_origin_split(dataset)
    assert train["target_timestamp"].max() <= validation["forecast_origin"].min()
    assert validation["target_timestamp"].max() <= test["forecast_origin"].min()
