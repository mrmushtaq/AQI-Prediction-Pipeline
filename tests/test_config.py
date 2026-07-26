"""Unit tests for `configs.config`, covering missing/invalid environment
variables (Step 1 test matrix: "Missing .env values" -> ConfigurationError).
"""

from __future__ import annotations

import pytest

from configs.config import load_settings
from src.common.exceptions import ConfigurationError

REQUIRED_ENV_VARS = {
    "AQICN_TOKEN": "test-aqicn-token",
    "OPENWEATHER_API_KEY": "test-openweather-key",
    "OPENAQ_API_KEY": "test-openaq-key",
    "DEFAULT_CITY": "Sukkur",
    "OPENWEATHER_LAT": "27.7052",
    "OPENWEATHER_LON": "68.8574",
}


def _set_all_env_vars(monkeypatch, overrides: dict | None = None) -> None:
    """Set all required env vars, then apply any overrides (None = unset)."""
    values = {**REQUIRED_ENV_VARS, **(overrides or {})}
    for key, value in values.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_load_settings_succeeds_with_all_required_vars(monkeypatch):
    _set_all_env_vars(monkeypatch)
    settings = load_settings()

    assert settings.aqicn_token == "test-aqicn-token"
    assert settings.default_city == "Sukkur"
    assert settings.latitude == pytest.approx(27.7052)
    assert settings.longitude == pytest.approx(68.8574)


@pytest.mark.parametrize("missing_var", list(REQUIRED_ENV_VARS.keys()))
def test_load_settings_raises_configuration_error_when_required_var_missing(monkeypatch, missing_var):
    _set_all_env_vars(monkeypatch, overrides={missing_var: None})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_raises_configuration_error_when_required_var_empty(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"AQICN_TOKEN": "   "})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_raises_configuration_error_on_non_numeric_latitude(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LAT": "not-a-number"})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_raises_configuration_error_on_non_numeric_longitude(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LON": "not-a-number"})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_raises_configuration_error_on_out_of_range_latitude(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LAT": "95.0"})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_raises_configuration_error_on_out_of_range_longitude(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LON": "200.0"})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_defaults_log_level_when_not_set(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"LOG_LEVEL": None})
    settings = load_settings()
    assert settings.log_level == "INFO"


def test_load_settings_accepts_boundary_latitude_90(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LAT": "90.0"})
    settings = load_settings()
    assert settings.latitude == 90.0


def test_load_settings_accepts_boundary_latitude_negative_90(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LAT": "-90.0"})
    settings = load_settings()
    assert settings.latitude == -90.0


def test_load_settings_accepts_boundary_longitude_180(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LON": "180.0"})
    settings = load_settings()
    assert settings.longitude == 180.0


def test_load_settings_accepts_boundary_longitude_negative_180(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LON": "-180.0"})
    settings = load_settings()
    assert settings.longitude == -180.0


def test_load_settings_raises_when_latitude_exceeds_90(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LAT": "90.1"})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_raises_when_longitude_exceeds_180(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LON": "180.1"})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_accepts_valid_api_keys(monkeypatch):
    _set_all_env_vars(monkeypatch)
    settings = load_settings()
    assert settings.aqicn_token == "test-aqicn-token"
    assert settings.openweather_api_key == "test-openweather-key"
    assert settings.openaq_api_key == "test-openaq-key"


def test_load_settings_empty_string_required_var_strips_to_empty(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"AQICN_TOKEN": ""})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_whitespace_only_var_treated_as_missing(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"AQICN_TOKEN": "   "})
    with pytest.raises(ConfigurationError):
        load_settings()


def test_load_settings_latitude_zero_is_valid(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LAT": "0"})
    settings = load_settings()
    assert settings.latitude == 0.0


def test_load_settings_longitude_zero_is_valid(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"OPENWEATHER_LON": "0"})
    settings = load_settings()
    assert settings.longitude == 0.0


def test_load_settings_accepts_default_city_with_spaces(monkeypatch):
    _set_all_env_vars(monkeypatch, overrides={"DEFAULT_CITY": "New York"})
    settings = load_settings()
    assert settings.default_city == "New York"
