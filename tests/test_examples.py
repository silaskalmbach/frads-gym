"""The example scripts run end to end with EnergyPlus and Radiance and give plausible physics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.simulation]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
DAY = {"begin_year": 2002, "begin_month": 6, "begin_day": 21, "end_year": 2002, "end_month": 6, "end_day": 21}


@pytest.fixture(scope="module")
def quickstart(tmp_path_factory):
    import quickstart

    quickstart.HERE = tmp_path_factory.mktemp("examples")  # glazing and outputs go here
    return quickstart


def _run(env, policy, key):
    env.reset()
    values = []
    for step in range(10_000):
        _, _, terminated, truncated, info = env.step(np.array([policy(step / 4)], dtype=np.float32))
        if terminated or truncated:
            return values
        values.append(float(np.ravel(info[f"raw_next_{key}"])[0]))


def test_quickstart_day(quickstart):
    env = quickstart.make_env(quickstart.HERE / "quickstart", run_period=DAY)
    try:
        wpi = _run(env, lambda hour: 0.9 if 11 <= hour < 15 else 0.1, "wpi")
    finally:
        env.close()
    assert len(wpi) == 95  # reset() consumes the first of 96 timesteps (sync_first_action=False)
    assert wpi[0] == 0.0 and wpi[-1] == 0.0  # night
    assert wpi[4 * 10] > 1000  # clear glazing at 10:00
    assert max(wpi[4 * 11 + 1:4 * 15]) < 0.1 * wpi[4 * 10]  # dark glazing around noon


def test_dark_glazing_lowers_cooling(quickstart):
    config = quickstart.make_config(radiance=False)
    config["cooling"] = {"get_variable_value": {"name": "Zone Air System Sensible Cooling Energy",
                                                "key": quickstart.ZONE},
                         "observation_space": {"low": 0, "high": 1e9}}
    from frads_gym import FradsEnv

    env = FradsEnv(output_dir=str(quickstart.HERE / "eplus"), config_file=config,
                   weather_files_path=[quickstart.WEATHER], number_of_timesteps_per_hour=4,
                   enable_radiance=False, reward_function=lambda *a: 0.0, run_period=DAY)
    try:
        clear, dark = (sum(_run(env, lambda hour, a=a: a, "cooling")) for a in (0.1, 0.9))
    finally:
        env.close()
    assert 0 < dark < clear
