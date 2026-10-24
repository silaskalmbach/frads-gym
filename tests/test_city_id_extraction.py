"""Tests for FradsEnv.extract_city_id — the city identifier helper used to
match training trackers to eval envs (per-city normalizer transfer).
"""
import os
import sys
import pytest


from frads_gym.envs.frads_gym import FradsEnv


class TestExtractCityIdFromWeather:
    """Weather path is the preferred source — tested first."""

    def test_chicago_from_weather_path(self):
        paths = ["/abs/weather/TMY/Chicago/USA_IL_Chicago.OHare.Intl.AP.725300_TMYx_2year.epw"]
        assert FradsEnv.extract_city_id(config_file=None, weather_files_path=paths) == "chicago"

    def test_san_francisco_from_weather_path(self):
        paths = ["/abs/weather/TMY/San_Francisco/USA_CA_San.Francisco.Intl.AP.724940_TMYx_2year.epw"]
        assert FradsEnv.extract_city_id(config_file=None, weather_files_path=paths) == "san_francisco"

    def test_phoenix_from_weather_path(self):
        paths = ["/abs/weather/TMY/Phoenix/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMYx_2year.epw"]
        assert FradsEnv.extract_city_id(config_file=None, weather_files_path=paths) == "phoenix"

    def test_duluth_from_weather_path(self):
        paths = ["/abs/weather/TMY/Duluth/USA_MN_Duluth.Intl.AP.727450_TMYx_2year.epw"]
        assert FradsEnv.extract_city_id(config_file=None, weather_files_path=paths) == "duluth"

    def test_houston_from_weather_path(self):
        paths = ["/abs/weather/TMY/Houston/USA_TX_Houston-Bush.Intercontinental.AP.722430_TMYx_2year.epw"]
        assert FradsEnv.extract_city_id(config_file=None, weather_files_path=paths) == "houston"

    def test_case_is_normalized_to_lowercase(self):
        paths = ["/abs/weather/TMY/CHICAGO/file.epw"]
        assert FradsEnv.extract_city_id(config_file=None, weather_files_path=paths) == "chicago"


class TestExtractCityIdFromIdfFallback:
    """If weather_files_path is missing/empty, fall back to IDF model_path."""

    def test_idf_chicago(self):
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_5A_USA_IL_CHICAGO.idf"}}
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=None) == "chicago"

    def test_idf_san_francisco(self):
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_3C_USA_CA_SAN_FRANCISCO.idf"}}
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=None) == "san_francisco"

    def test_idf_houston(self):
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_2A_USA_TX_HOUSTON.idf"}}
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=None) == "houston"

    def test_idf_with_mod_suffix(self):
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_2B_USA_AZ_PHOENIX_MOD.idf"}}
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=None) == "phoenix"

    def test_weather_path_takes_priority_over_idf(self):
        """If both are provided, weather path wins."""
        config = {"model_setup": {"model_path": "..._CHICAGO.idf"}}
        paths = ["/abs/weather/TMY/Duluth/file.epw"]
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=paths) == "duluth"


class TestExtractCityIdErrors:
    def test_both_missing_raises(self):
        with pytest.raises(ValueError, match="Cannot extract city_id"):
            FradsEnv.extract_city_id(config_file=None, weather_files_path=None)

    def test_empty_weather_list_falls_back(self):
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_7A_USA_MN_DULUTH.idf"}}
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=[]) == "duluth"

    def test_idf_without_city_marker_raises(self):
        """IDF filename without USA_<state>_<CITY> marker must error."""
        config = {"model_setup": {"model_path": "random_office_building.idf"}}
        with pytest.raises(ValueError, match="Cannot extract city_id"):
            FradsEnv.extract_city_id(config_file=config, weather_files_path=None)

    def test_empty_weather_and_unparseable_idf_raises(self):
        config = {"model_setup": {"model_path": "something.idf"}}
        with pytest.raises(ValueError, match="Cannot extract city_id"):
            FradsEnv.extract_city_id(config_file=config, weather_files_path=[])

    def test_weather_path_with_trailing_slash_falls_through_to_idf(self):
        """Trailing slash lifts parent to 'tmy' — guardrail rejects, IDF wins."""
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_5A_USA_IL_CHICAGO.idf"}}
        paths = ["/abs/weather/TMY/Chicago/"]
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=paths) == "chicago"

    def test_weather_path_with_trailing_slash_and_no_idf_raises(self):
        """Trailing slash + no IDF fallback must raise, not silently return 'tmy'."""
        with pytest.raises(ValueError, match="Cannot extract city_id"):
            FradsEnv.extract_city_id(config_file=None, weather_files_path=["/abs/weather/TMY/Chicago/"])

    def test_weather_bare_filename_falls_through_to_idf(self):
        """Bare filename has no meaningful parent dir — guardrail rejects, IDF wins."""
        config = {"model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_7A_USA_MN_DULUTH.idf"}}
        paths = ["file.epw"]
        assert FradsEnv.extract_city_id(config_file=config, weather_files_path=paths) == "duluth"


class TestExtractCityIdWithStringConfigPath:
    """Exercise the branch where config_file is a path to a JSON file."""

    def test_idf_extraction_from_json_config_path(self, tmp_path):
        import json
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "model_setup": {"model_path": "RefBldgMediumOfficeNew2004_v1.4_7.2_5A_USA_IL_CHICAGO.idf"}
        }))
        assert FradsEnv.extract_city_id(config_file=str(config_path), weather_files_path=None) == "chicago"
