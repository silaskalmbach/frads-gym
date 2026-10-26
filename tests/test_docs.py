"""The configuration reference names every configuration key the code reads."""

from __future__ import annotations

import re
from pathlib import Path

from frads_gym import validation

ROOT = Path(__file__).resolve().parents[1]
# variables that hold (parts of) the configuration in the package code
CONFIG_VARS = ("ct|illum_cfg|cfg|lighting_calc|alloc_cfg|gv|edgp_cfg|model_setup|entry|lighting|glazing|var_info|"
               "cal|ev_cal|norm_config|standardize_config|lp|ms|state|s|c|g|v|config")
# keys of internal state read through the same variable names, not configuration
NOT_CONFIG = {"count", "mean", "std", "var", "p_low", "p_high", "error", "datetime", "day_of_week", "version",
              "simulation_finished", "zones", "model_sha256", "npz_sha256", "sensors_sha256", "states",
              "agent_active", "ema_quantile_trackers", "ema_trackers", "obs_history", "obs_history_dynamic",
              "obs_welford_trackers", "stat_trackers"}
# keys that need the frads fork; not documented in this release
FORK_ONLY = {"calculate_dgp", "ev_sensor", "save_hdr", "cfs_thermal", "cfs_thermal_systems", "equipment_name",
             "fraction_radiant", "windows", "state_to_gs", "pane_overrides", "h_out", "h_in", "gap_nu",
             "absorptance_mode", "log_pane_temps"}


def test_configuration_reference_is_complete():
    files = ("envs/frads_wrapper.py", "envs/frads_gym.py", "validation.py")  # where the config is read
    code = "".join((ROOT / "frads_gym" / f).read_text() for f in files)
    read = set(re.findall(rf"""(?:{CONFIG_VARS})(?:\.get\(|\[)\s*["']([a-z][a-z0-9_]+)["']""", code))
    read |= set(validation.SOURCES) | validation.ENTRY_KEYS | validation.MODEL_SETUP_KEYS
    doc = (ROOT / "docs" / "configuration.md").read_text()
    missing = sorted(k for k in read - NOT_CONFIG - FORK_ONLY if f"`{k}`" not in doc and f'"{k}"' not in doc)
    assert not missing, f"keys read by the code but missing in docs/configuration.md: {missing}"
