"""Invalid configurations and arguments fail at construction with a message naming the problem."""

from __future__ import annotations

import copy

import pytest

from frads_gym.validation import ConfigError, validate_config, validate_env_args

WINDOW = "Perimeter_mid_ZN_1_Wall_South_Window"
VALID = {
    "_comment": "keys starting with an underscore are ignored",
    "config_dir": "/tmp",
    "model_setup": {"model_type": "medium_office", "glazing_systems": [{"file": "a.json"}, {"file": "b.json"}]},
    "facade_state_mapping": {"states": [{"name": "a", "max_threshold": 0.5}, {"name": "b", "max_threshold": 1.0}]},
    "cfs_state": {"get_cfs_state": {"key": WINDOW}, "actuate_cfs_state": {"key": WINDOW},
                  "observation_space": {"low": 0, "high": 1}, "action_space": {"low": 0, "high": 1}},
    "t_out": {"get_variable_value": {"name": "Site Outdoor Air Drybulb Temperature", "key": "Environment"},
              "observation_space": {"low": -1, "high": 1},
              "observation_normalize": {"active": True, "dynamic": False, "low": -20, "high": 40}},
    "wpi": {"calculate_illuminance": {"zone": "Perimeter_mid_ZN_1", "cfs_name": [WINDOW]},
            "post_processing": "mean", "observation_space": {"low": 0, "high": 1},
            "calibration": {"method": "origin_scaling", "k": 1.1}},
    "lights": {"lighting_power_calculation": {"power_density": 1000, "lux_threshold": 500}, "depends_on": "wpi",
               "observation_space": {"active": False}},
    "total": {"calculate_energy_sum": {"keys_to_sum": ["lights"]}, "observation_space": {"low": 0, "high": 1}},
    "off": {"active": False, "anything": "is not checked"},
}


def broken(change):
    cfg = copy.deepcopy(VALID)
    change(cfg)
    return cfg


def test_valid_config_passes():
    validate_config(VALID)


@pytest.mark.parametrize("change, message", [
    (lambda c: c.pop("model_setup"), "'model_setup' is missing"),
    (lambda c: c["model_setup"].update(model_type="skyscraper"), "needs 'model_path' or a 'model_type'"),
    (lambda c: c["model_setup"].update(glazing="x"), "'model_setup.glazing' is unknown"),
    (lambda c: c["t_out"].update(get_variable="x"), "'t_out.get_variable' is unknown"),
    (lambda c: c.update(old={"calculate_wpi": {}, "observation_space": {"low": 0, "high": 1}}),
     "'old.calculate_wpi' is no longer supported"),
    (lambda c: c.update(empty={"observation_space": {"low": 0, "high": 1}}), "'empty' has no data source"),
    (lambda c: c["t_out"].update(get_cfs_state={"key": WINDOW}), "more than one data source"),
    (lambda c: c["cfs_state"].pop("action_space"), "needs both 'actuate_cfs_state' and 'action_space'"),
    (lambda c: c["t_out"]["observation_space"].pop("high"), "'t_out.observation_space' needs 'low' and 'high'"),
    (lambda c: c.update(tint=c.pop("cfs_state")), "must contain 'cfs_state'"),
    (lambda c: c["lights"].update(depends_on="nothing"), "refers to 'nothing'"),
    (lambda c: c["total"]["calculate_energy_sum"].update(keys_to_sum=["off"]), "refers to 'off'"),
    (lambda c: c["facade_state_mapping"]["states"].reverse(), "thresholds must increase"),
    (lambda c: c["facade_state_mapping"]["states"][1].update(max_threshold=0.9), "end at >= 1"),
    (lambda c: c["facade_state_mapping"]["states"][1].update(name="a"), "duplicate state names"),
    (lambda c: c["wpi"]["calibration"].update(k="1.1"), "'wpi.calibration' must be"),
    (lambda c: c.update(note="text"), "'note' must be an object"),
])
def test_invalid_config_is_named(change, message):
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        validate_config(broken(change))


def test_all_errors_are_listed_at_once():
    cfg = broken(lambda c: (c["t_out"].update(typo=1), c["wpi"].update(typo=2)))
    with pytest.raises(ConfigError) as e:
        validate_config(cfg)
    assert "'t_out.typo'" in str(e.value) and "'wpi.typo'" in str(e.value)


def test_dgp_may_come_with_edgps():
    cfg = broken(lambda c: c.update(glare={"calculate_dgp": {"zone": "z", "cfs_name": [WINDOW]},
                                           "calculate_edgps": {"zone": "z", "cfs_name": [WINDOW]},
                                           "observation_space": {"low": 0, "high": 1}}))
    validate_config(cfg)


@pytest.fixture
def args(tmp_path):
    epw = tmp_path / "w.epw"
    epw.write_text("")
    return dict(config_file=VALID, output_dir=str(tmp_path), reward_function=lambda *a: 0.0,
                weather_files_path=[str(epw)], number_of_timesteps_per_hour=4, truncate_on="none",
                run_period=None, run_periods=None, staggered_start=False, env_id=0, n_envs=1)


def test_valid_args_pass(args):
    validate_env_args(**args)


@pytest.mark.parametrize("change, message", [
    (dict(config_file=None), "config_file is required"),
    (dict(config_file="/no/such/config.json"), "config_file not found"),
    (dict(output_dir=None), "output_dir is required"),
    (dict(reward_function=None), "reward_function is required"),
    (dict(weather_files_path=None), "weather_files_path is required"),
    (dict(weather_files_path=["/no/such.epw"]), "weather file not found"),
    (dict(number_of_timesteps_per_hour=7), "number_of_timesteps_per_hour must be one of"),
    (dict(truncate_on="week"), "truncate_on must be one of"),
    (dict(run_period={"begin_monht": 1}), "unknown keys"),
    (dict(run_periods=[]), "run_periods has 0 entries for 1 weather files"),
    (dict(staggered_start=True), "staggered_start needs run_periods"),
    (dict(env_id=4, n_envs=4), "env_id must be in"),
])
def test_invalid_args_are_named(args, change, message):
    with pytest.raises(ConfigError, match=message):
        validate_env_args(**{**args, **change})
