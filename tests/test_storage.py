"""Unit tests for `src.utils.storage`."""

from __future__ import annotations

import pytest

from src.common.exceptions import StorageError
from src.utils.storage import load_csv, load_json, save_csv, save_json, save_metadata


def test_save_and_load_json_roundtrip(tmp_path):
    data = {"city": "Sukkur", "aqi": 152, "components": {"pm2_5": 90}}
    path = tmp_path / "nested" / "sample.json"

    saved_path = save_json(data, path)
    assert saved_path.exists()

    loaded = load_json(path)
    assert loaded == data


def test_save_json_creates_missing_directories(tmp_path):
    path = tmp_path / "a" / "b" / "c" / "sample.json"
    save_json({"x": 1}, path)
    assert path.exists()


def test_load_json_missing_file_raises_storage_error(tmp_path):
    with pytest.raises(StorageError):
        load_json(tmp_path / "does_not_exist.json")


def test_save_and_load_csv_roundtrip(tmp_path):
    rows = [
        {"city": "Sukkur", "aqi": "152"},
        {"city": "Karachi", "aqi": "180"},
    ]
    path = tmp_path / "processed" / "dataset.csv"

    save_csv(rows, path)
    loaded = load_csv(path)

    assert loaded == rows


def test_save_csv_appends_on_subsequent_calls(tmp_path):
    path = tmp_path / "dataset.csv"
    save_csv([{"city": "Sukkur", "aqi": "152"}], path)
    save_csv([{"city": "Karachi", "aqi": "180"}], path)

    loaded = load_csv(path)
    assert len(loaded) == 2
    assert loaded[0]["city"] == "Sukkur"
    assert loaded[1]["city"] == "Karachi"


def test_save_csv_raises_on_empty_rows(tmp_path):
    with pytest.raises(StorageError):
        save_csv([], tmp_path / "empty.csv")


def test_save_csv_raises_on_inconsistent_columns(tmp_path):
    rows = [{"city": "Sukkur", "aqi": "152"}, {"city": "Karachi"}]
    with pytest.raises(StorageError):
        save_csv(rows, tmp_path / "bad.csv")


def test_load_csv_missing_file_raises_storage_error(tmp_path):
    with pytest.raises(StorageError):
        load_csv(tmp_path / "missing.csv")


def test_save_metadata_with_explicit_path(tmp_path):
    meta = {"pipeline_type": "live", "records": 1, "status": "SUCCESS"}
    path = tmp_path / "meta_test.json"
    saved = save_metadata(meta, pipeline_type="live", filepath=path)
    assert saved.exists()
    loaded = load_json(path)
    assert loaded["pipeline_type"] == "live"
    assert loaded["status"] == "SUCCESS"


def test_save_metadata_generates_path(tmp_path):
    meta = {"pipeline_type": "historical", "records": 10, "status": "SUCCESS"}
    saved = save_metadata(meta, pipeline_type="historical")
    assert saved.exists()
    assert "metadata_historical_" in saved.name
    assert saved.suffix == ".json"
