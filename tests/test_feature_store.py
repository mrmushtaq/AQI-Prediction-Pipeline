"""Unit tests for `src.feature_pipeline.feature_store`.

All Hopsworks interaction is mocked -- these tests never require a real
Hopsworks account, project, or API key.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from configs.config import HopsworksSettings
from src.common.exceptions import FeatureStoreError
from src.feature_pipeline import feature_store


@pytest.fixture
def hopsworks_settings() -> HopsworksSettings:
    return HopsworksSettings(api_key="test-api-key", project_name="test_project")


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-07-26T09:00:00+00:00", "2026-07-26T12:00:00+00:00"],
            "collection_timestamp": ["2026-07-26T09:00:00+00:00", "2026-07-26T12:00:00+00:00"],
            "city": ["Sukkur", "Sukkur"],
            "aqi": [156, 140],
            "pm2_5": [156, 130],
            "aqi_target": [120, 110],
        }
    )


# --- prepare_dataframe_for_feature_store -----------------------------------


def test_prepare_dataframe_converts_collection_timestamp_to_datetime(sample_feature_df):
    prepared = feature_store.prepare_dataframe_for_feature_store(sample_feature_df)
    assert pd.api.types.is_datetime64_any_dtype(prepared["collection_timestamp"])


def test_prepare_dataframe_drops_redundant_timestamp_column(sample_feature_df):
    prepared = feature_store.prepare_dataframe_for_feature_store(sample_feature_df)
    assert "timestamp" not in prepared.columns
    assert "collection_timestamp" in prepared.columns


def test_prepare_dataframe_does_not_mutate_original(sample_feature_df):
    original_columns = list(sample_feature_df.columns)
    feature_store.prepare_dataframe_for_feature_store(sample_feature_df)
    assert list(sample_feature_df.columns) == original_columns
    # The original dataframe's collection_timestamp must remain a plain
    # string column (untouched) -- only the returned copy is converted to
    # datetime64.
    assert not pd.api.types.is_datetime64_any_dtype(sample_feature_df["collection_timestamp"])


def test_prepare_dataframe_raises_when_collection_timestamp_missing():
    df = pd.DataFrame({"city": ["Sukkur"], "aqi": [100]})
    with pytest.raises(FeatureStoreError):
        feature_store.prepare_dataframe_for_feature_store(df)


def test_prepare_dataframe_raises_on_unparseable_timestamp():
    df = pd.DataFrame({"collection_timestamp": ["not-a-date"], "city": ["Sukkur"]})
    with pytest.raises(FeatureStoreError):
        feature_store.prepare_dataframe_for_feature_store(df)


# --- login ------------------------------------------------------------------


def test_login_raises_when_neither_hopsworks_nor_hsfs_installed(hopsworks_settings):
    with patch.dict(sys.modules, {"hopsworks": None, "hsfs": None}):
        with pytest.raises(FeatureStoreError, match="Neither 'hopsworks' nor 'hsfs'"):
            feature_store.login(hopsworks_settings)


def test_login_falls_back_to_hsfs_when_hopsworks_not_installed(hopsworks_settings):
    """When the full `hopsworks` package isn't installed (e.g. Windows
    without a C compiler for the twofish/pyjks dependency), `login()` must
    transparently fall back to the lighter `hsfs`-only client.
    """
    fake_fs = MagicMock(name="feature_store_handle")
    fake_connection = MagicMock(name="hsfs_connection")
    fake_connection.get_feature_store.return_value = fake_fs

    fake_hsfs_module = ModuleType("hsfs")
    fake_hsfs_module.connection = MagicMock(return_value=fake_connection)

    with patch.dict(sys.modules, {"hopsworks": None, "hsfs": fake_hsfs_module}):
        result = feature_store.login(hopsworks_settings)

    fake_hsfs_module.connection.assert_called_once_with(
        host=feature_store.DEFAULT_HOPSWORKS_HOST,
        project="test_project",
        api_key_value="test-api-key",
    )
    assert result is fake_fs


def test_login_hsfs_fallback_wraps_connection_errors(hopsworks_settings):
    fake_hsfs_module = ModuleType("hsfs")
    fake_hsfs_module.connection = MagicMock(side_effect=Exception("bad host"))

    with patch.dict(sys.modules, {"hopsworks": None, "hsfs": fake_hsfs_module}):
        with pytest.raises(FeatureStoreError, match="hsfs connection failed"):
            feature_store.login(hopsworks_settings)


def test_login_success_returns_feature_store_handle(hopsworks_settings):
    fake_fs = MagicMock(name="feature_store_handle")
    fake_project = MagicMock(name="project")
    fake_project.get_feature_store.return_value = fake_fs

    fake_hopsworks_module = ModuleType("hopsworks")
    fake_hopsworks_module.login = MagicMock(return_value=fake_project)

    with patch.dict(sys.modules, {"hopsworks": fake_hopsworks_module}):
        result = feature_store.login(hopsworks_settings)

    fake_hopsworks_module.login.assert_called_once_with(
        project="test_project", api_key_value="test-api-key"
    )
    assert result is fake_fs


def test_login_raises_feature_store_error_on_login_failure(hopsworks_settings):
    fake_hopsworks_module = ModuleType("hopsworks")
    fake_hopsworks_module.login = MagicMock(side_effect=Exception("invalid api key"))

    with patch.dict(sys.modules, {"hopsworks": fake_hopsworks_module}):
        with pytest.raises(FeatureStoreError, match="login failed"):
            feature_store.login(hopsworks_settings)


# --- get_or_create_feature_group / insert_features / read_features ---------


def test_get_or_create_feature_group_uses_expected_schema():
    fake_fs = MagicMock()
    fake_fg = MagicMock()
    fake_fs.get_or_create_feature_group.return_value = fake_fg

    result = feature_store.get_or_create_feature_group(fake_fs)

    fake_fs.get_or_create_feature_group.assert_called_once_with(
        name="aqi_features",
        version=1,
        description=feature_store.FEATURE_GROUP_DESCRIPTION,
        primary_key=["city", "collection_timestamp"],
        event_time="collection_timestamp",
        online_enabled=False,
    )
    assert result is fake_fg


def test_get_or_create_feature_group_wraps_errors():
    fake_fs = MagicMock()
    fake_fs.get_or_create_feature_group.side_effect = Exception("connection reset")

    with pytest.raises(FeatureStoreError):
        feature_store.get_or_create_feature_group(fake_fs)


def test_insert_features_calls_fg_insert(sample_feature_df):
    fake_fg = MagicMock()
    feature_store.insert_features(fake_fg, sample_feature_df)
    fake_fg.insert.assert_called_once_with(sample_feature_df)


def test_insert_features_wraps_errors(sample_feature_df):
    fake_fg = MagicMock()
    fake_fg.insert.side_effect = Exception("schema mismatch")
    with pytest.raises(FeatureStoreError):
        feature_store.insert_features(fake_fg, sample_feature_df)


def test_read_features_returns_dataframe(sample_feature_df):
    fake_fg = MagicMock()
    fake_fg.read.return_value = sample_feature_df
    result = feature_store.read_features(fake_fg)
    assert result is sample_feature_df


def test_read_features_wraps_errors():
    fake_fg = MagicMock()
    fake_fg.read.side_effect = Exception("read timeout")
    with pytest.raises(FeatureStoreError):
        feature_store.read_features(fake_fg)


# --- sync_feature_store (end-to-end, fully mocked) --------------------------


def test_sync_feature_store_raises_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FeatureStoreError, match="not found"):
        feature_store.sync_feature_store(missing_path)


def test_sync_feature_store_end_to_end(tmp_path, sample_feature_df):
    csv_path = tmp_path / "feature_df.csv"
    sample_feature_df.to_csv(csv_path, index=False)

    fake_fs = MagicMock()
    fake_fg = MagicMock()
    fake_fg.read.return_value = sample_feature_df

    with patch.object(feature_store, "login", return_value=fake_fs) as mock_login, patch.object(
        feature_store, "get_or_create_feature_group", return_value=fake_fg
    ) as mock_get_fg:
        result = feature_store.sync_feature_store(csv_path)

    mock_login.assert_called_once()
    mock_get_fg.assert_called_once_with(fake_fs)
    fake_fg.insert.assert_called_once()
    fake_fg.read.assert_called_once()
    assert result is sample_feature_df