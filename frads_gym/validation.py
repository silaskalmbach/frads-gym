"""Checks of the configuration and the FradsEnv arguments.

Mistakes fail when the environment is created, with every problem listed at once, instead of
surfacing as silent zeros or a hang somewhere in a simulation.
"""

import os
from numbers import Real

# keys that produce an observation; every active entry needs exactly one (or only actuate_cfs_state),
# except that calculate_dgp may come with calculate_edgps
SOURCES = ("get_variable_value", "get_cfs_state", "calculate_illuminance", "calculate_edgps", "calculate_dgp",
           "lighting_power_calculation", "calculate_allocation", "calculate_energy_sum", "cfs_thermal")
ENTRY_KEYS = set(SOURCES) | {
    "active", "observation_space", "action_space", "observation_normalize", "observation_standardize",
    "actuate_cfs_state", "post_processing", "depends_on", "divider", "calibration", "hvac_substep_accumulate",
    "log_pane_temps"}
MODEL_SETUP_KEYS = {"model_path", "model_type", "lighting_systems", "glazing_systems", "cfs_thermal_systems",
                    "fixed_matrices"}
TIMESTEPS_PER_HOUR = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)  # divisors of 60 that EnergyPlus accepts
TRUNCATE_ON = ("none", "day", "month")
RUN_PERIOD_KEYS = {"begin_year", "begin_month", "begin_day", "end_year", "end_month", "end_day"}
REMOVED = {"calculate_wpi": "use 'calculate_illuminance' with \"post_processing\": \"mean\""}


class ConfigError(ValueError):
    """An invalid frads-gym configuration or FradsEnv argument."""


def _raise(errors, what):
    if errors:
        raise ConfigError(f"Invalid {what}:\n  - " + "\n  - ".join(errors))


def _is_meta(key):
    return key.startswith("_") or key in ("config_dir", "config_file_path")


def _bounds(entry_id, name, spec, errors):
    if not isinstance(spec, dict):
        errors.append(f"'{entry_id}.{name}' must be an object")
    elif spec.get("active", True) and not ("low" in spec and "high" in spec):
        errors.append(f"'{entry_id}.{name}' needs 'low' and 'high'")


def validate_config(config):
    """Raise ConfigError listing every problem of a frads-gym configuration dict."""
    if not isinstance(config, dict):
        raise ConfigError(f"Invalid configuration: expected a dict, got {type(config).__name__}")
    errors = []
    entries = {k: v for k, v in config.items() if not _is_meta(k) and k not in ("model_setup", "facade_state_mapping")}
    for k, v in entries.items():
        if not isinstance(v, dict):
            errors.append(f"'{k}' must be an object (prefix it with '_' for comments)")
    active = {k: v for k, v in entries.items() if isinstance(v, dict) and v.get("active", True)}

    ms = config.get("model_setup")
    if not isinstance(ms, dict):
        errors.append("'model_setup' is missing")
    else:
        errors += [f"'model_setup.{k}' is unknown" for k in ms if not _is_meta(k) and k not in MODEL_SETUP_KEYS]
        if not ms.get("model_path"):
            from pyenergyplus.dataset import ref_models
            if ms.get("model_type") not in ref_models:
                errors.append(f"'model_setup' needs 'model_path' or a 'model_type' out of {sorted(ref_models)}")
        for i, g in enumerate(ms.get("glazing_systems", [])):
            if not (isinstance(g, dict) and isinstance(g.get("file"), str)):
                errors.append(f"'model_setup.glazing_systems[{i}]' needs a 'file'")
        for i, c in enumerate(ms.get("cfs_thermal_systems", [])):
            if not (isinstance(c, dict) and "equipment_name" in c and "zone" in c):
                errors.append(f"'model_setup.cfs_thermal_systems[{i}]' needs 'equipment_name' and 'zone'")

    if any("get_cfs_state" in v or "actuate_cfs_state" in v for v in active.values()):
        states = (config.get("facade_state_mapping") or {}).get("states")
        if not (isinstance(states, list) and states and all(isinstance(s, dict) for s in states)):
            errors.append("'facade_state_mapping.states' must be a non-empty list of {name, max_threshold}")
        elif not all(isinstance(s.get("name"), str) and isinstance(s.get("max_threshold"), Real) for s in states):
            errors.append("every 'facade_state_mapping' state needs a 'name' and a numeric 'max_threshold'")
        else:
            names = [s["name"] for s in states]
            limits = [s["max_threshold"] for s in states]
            if len(set(names)) != len(names):
                errors.append(f"'facade_state_mapping' has duplicate state names: {names}")
            if any(b <= a for a, b in zip(limits, limits[1:])) or limits[-1] < 1:
                errors.append(f"'facade_state_mapping' thresholds must increase and end at >= 1, got {limits}")

    for k, v in active.items():
        for key in v:
            if key in REMOVED:
                errors.append(f"'{k}.{key}' is no longer supported: {REMOVED[key]}")
            elif not _is_meta(key) and key not in ENTRY_KEYS:
                errors.append(f"'{k}.{key}' is unknown")
        # calculate_dgp takes precedence over calculate_edgps in the same entry (eDGP stays as documentation)
        sources = [s for s in SOURCES if s in v and not (s == "calculate_edgps" and "calculate_dgp" in v)]
        if len(sources) > 1:
            errors.append(f"'{k}' has more than one data source: {sources}")
        elif not sources and "actuate_cfs_state" not in v and not any(key in REMOVED for key in v):
            errors.append(f"'{k}' has no data source (one of {', '.join(SOURCES)})")
        if ("action_space" in v) != ("actuate_cfs_state" in v):
            errors.append(f"'{k}' needs both 'actuate_cfs_state' and 'action_space'")
        for name in ("observation_space", "action_space"):
            if name in v:
                _bounds(k, name, v[name], errors)
        for name in ("observation_normalize",):
            if isinstance(v.get(name), dict) and v[name].get("active"):
                _bounds(k, name, v[name], errors)
        if ("get_cfs_state" in v and "cfs_state" not in k
                and v.get("observation_space", {}).get("active", True) and "observation_space" in v):
            errors.append(f"'{k}': the key of an observed get_cfs_state entry must contain 'cfs_state'")
        cal = v.get("calibration")
        if cal is not None and not (isinstance(cal, dict) and cal.get("method") == "origin_scaling"
                                    and isinstance(cal.get("k", 1.0), Real)):
            errors.append(f"'{k}.calibration' must be {{\"method\": \"origin_scaling\", \"k\": <number>}}")
        lp = v.get("lighting_power_calculation")
        if lp is not None:
            dep = active.get(v.get("depends_on"), {})
            if "calculate_illuminance" not in dep:
                errors.append(f"'{k}': 'lighting_power_calculation' needs 'depends_on' naming an active "
                              "calculate_illuminance entry")
            if not (isinstance(lp, dict) and "lux_threshold" in lp
                    and ("power_density" in lp or {"power_density_m2", "area_m2"} <= set(lp))):
                errors.append(f"'{k}.lighting_power_calculation' needs 'lux_threshold' and 'power_density' "
                              "[W] or 'power_density_m2' [W/m2] with 'area_m2'")
            if isinstance(lp, dict) and lp.get("occupancy_key") is not None:
                refs_extra = [lp["occupancy_key"]]
            else:
                refs_extra = []
        else:
            refs_extra = []
        refs = [v.get("depends_on")]
        refs += (v.get("calculate_energy_sum") or {}).get("keys_to_sum", [])
        refs += [(v.get("calculate_allocation") or {}).get(x)
                 for x in ("source_key", "numerator_key", "denominator_key")]
        refs.append((v.get("calculate_dgp") or {}).get("ev_sensor"))
        refs += refs_extra
        errors += [f"'{k}' refers to '{r}', which is not an active entry"
                   for r in refs if r is not None and r not in active]
    _raise(errors, "configuration")


def validate_env_args(*, config_file, output_dir, reward_function, weather_files_path, number_of_timesteps_per_hour,
                      truncate_on, run_period, run_periods, staggered_start, env_id, n_envs):
    """Raise ConfigError listing every problem of the FradsEnv constructor arguments."""
    errors = []
    if config_file is None:
        errors.append("config_file is required (a path to a JSON file or a dict)")
    elif isinstance(config_file, str) and not os.path.isfile(config_file):
        errors.append(f"config_file not found: {config_file}")
    elif not isinstance(config_file, (str, dict)):
        errors.append(f"config_file must be a path or a dict, got {type(config_file).__name__}")
    if output_dir is None:
        errors.append("output_dir is required")
    if not callable(reward_function):
        errors.append("reward_function is required: a callable (obs, action, next_obs, info) -> float")
    if not (isinstance(weather_files_path, (list, tuple)) and weather_files_path):
        errors.append("weather_files_path is required: a non-empty list of EPW files")
    else:
        errors += [f"weather file not found: {w}" for w in weather_files_path if not os.path.isfile(w)]
    if number_of_timesteps_per_hour not in TIMESTEPS_PER_HOUR:
        errors.append(f"number_of_timesteps_per_hour must be one of {TIMESTEPS_PER_HOUR}, "
                      f"got {number_of_timesteps_per_hour!r}")
    if not (isinstance(truncate_on, str) and truncate_on.lower() in TRUNCATE_ON):
        errors.append(f"truncate_on must be one of {TRUNCATE_ON}, got {truncate_on!r}")
    periods = [("run_period", run_period)] if run_period is not None else []
    if run_periods is not None:
        if not isinstance(run_periods, (list, tuple)):
            errors.append("run_periods must be a list of run-period dicts")
            run_periods = []
        periods += [(f"run_periods[{i}]", p) for i, p in enumerate(run_periods)]
        if isinstance(weather_files_path, (list, tuple)) and len(run_periods) < len(weather_files_path):
            errors.append(f"run_periods has {len(run_periods)} entries for {len(weather_files_path)} weather files")
    for name, p in periods:
        if not isinstance(p, dict):
            errors.append(f"{name} must be a dict")
        elif set(p) - RUN_PERIOD_KEYS:
            errors.append(f"{name} has unknown keys {sorted(set(p) - RUN_PERIOD_KEYS)}; "
                          f"allowed: {sorted(RUN_PERIOD_KEYS)}")
    if staggered_start and not run_periods:
        errors.append("staggered_start needs run_periods")
    if not (isinstance(n_envs, int) and n_envs >= 1 and isinstance(env_id, int) and 0 <= env_id < n_envs):
        errors.append(f"env_id must be in [0, n_envs), got env_id={env_id!r}, n_envs={n_envs!r}")
    _raise(errors, "FradsEnv arguments")
