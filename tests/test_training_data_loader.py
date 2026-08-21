"""Unit tests for `src.training_pipeline.data_loader`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.common.exceptions import TrainingError
from src.training_pipeline.data_loader import (
    load_features,
    load_features_from_hopsworks,
    load_features_from_local_csv,
)


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "collection_timestamp": ["2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00"],
            "city": ["Sukkur", "Sukkur"],
            "aqi": [100, 110],
            "aqi_target": [120, 130],
        }
    )
    path = tmp_path / "feature_df.csv"
    df.to_csv(path, index=False)
    return path


def test_load_features_from_local_csv_success(sample_csv):
    df = load_features_from_local_csv(sample_csv)
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["collection_timestamp"])


def test_load_features_from_local_csv_raises_when_missing(tmp_path):
    with pytest.raises(TrainingError):
        load_features_from_local_csv(tmp_path / "does_not_exist.csv")


def test_load_features_dispatches_to_local(sample_csv):
    df = load_features(source="local", local_path=sample_csv)
    assert len(df) == 2


def test_load_features_raises_on_invalid_source():
    with pytest.raises(TrainingError):
        load_features(source="not_a_real_source")  # type: ignore[arg-type]


def test_load_features_from_hopsworks_wraps_feature_store_errors():
    from src.common.exceptions import FeatureStoreError

    with patch("src.feature_pipeline.feature_store.login", side_effect=FeatureStoreError("login failed")):
        with pytest.raises(TrainingError):
            load_features_from_hopsworks()


def test_load_features_from_hopsworks_success():
    sample_df = pd.DataFrame({"aqi": [1, 2], "aqi_target": [3, 4]})
    fake_fs = MagicMock()
    fake_fg = MagicMock()
    fake_fs.get_feature_group.return_value = fake_fg

    with patch("src.feature_pipeline.feature_store.login", return_value=fake_fs), patch(
        "src.feature_pipeline.feature_store.read_features", return_value=sample_df
    ):
        df = load_features_from_hopsworks()

    assert len(df) == 2
    fake_fs.get_feature_group.assert_called_once()
