"""Quickstart: one summer day in the DOE medium office (Chicago) with switchable glazing.

Everything comes from the installed packages: the building model and the weather file ship with
pyenergyplus, the four glazing states are created by make_glazing.py. The first run generates
the Radiance matrices, which takes a few minutes.

    python quickstart.py
"""

from pathlib import Path

import numpy as np
from pyenergyplus.dataset import weather_files

from frads_gym import FradsEnv
from make_glazing import STATES, make_glazing

HERE = Path(__file__).resolve().parent
ZONE = "Perimeter_mid_ZN_1"
WINDOW = "Perimeter_mid_ZN_1_Wall_South_Window"
WEATHER = weather_files["usa_il_chicago"]


def variable(name, key, high, low=0.0):
    """An EnergyPlus output variable, scaled from [low, high] to the observation range [0, 1]."""
    return {"get_variable_value": {"name": name, "key": key},
            "observation_space": {"low": 0, "high": 1},
            "observation_normalize": {"active": True, "dynamic": False, "low": low, "high": high}}


def make_config(radiance=True):
    glazing = make_glazing(HERE / "output" / "glazing")
    config = {
        "model_setup": {
            "model_type": "medium_office",  # pyenergyplus reference model, Chicago
            "glazing_systems": [{"file": str(p)} for p in glazing],
        },
        # action in [0, 1] -> glazing state, clear to dark
        "facade_state_mapping": {"states": [
            {"name": name, "max_threshold": (i + 1) / len(STATES)} for i, name in enumerate(STATES)]},
        # the key must contain "cfs_state": its value (a state name) is mapped back to [0, 1]
        "cfs_state": {"get_cfs_state": {"key": WINDOW}, "actuate_cfs_state": {"key": WINDOW},
                      "observation_space": {"low": 0, "high": 1}, "action_space": {"low": 0, "high": 1}},
        "outdoor_temp": variable("Site Outdoor Air Drybulb Temperature", "Environment", 40, -20),
        "indoor_temp": variable("Zone Mean Air Temperature", ZONE, 35, 15),
        "direct_normal_irradiance": variable("Site Direct Solar Radiation Rate per Area", "Environment", 1000),
        "diffuse_horizontal_irradiance": variable("Site Diffuse Solar Radiation Rate per Area", "Environment", 500),
    }
    if radiance:
        # mean work-plane illuminance of the zone [lx], computed with the three-phase method
        config["wpi"] = {"calculate_illuminance": {"zone": ZONE, "cfs_name": [WINDOW]},
                         "post_processing": "mean",
                         "observation_space": {"low": 0, "high": 1},
                         "observation_normalize": {"active": True, "dynamic": False, "low": 0, "high": 2000}}
    return config


def daylight_reward(obs, action, next_obs, info):
    """Toy reward: keep the work-plane illuminance close to 500 lx."""
    wpi = float(np.ravel(info["raw_next_wpi"])[0])
    return -abs(wpi - 500.0) / 500.0


def make_env(output_dir, radiance=True, **kwargs):
    return FradsEnv(
        output_dir=str(output_dir),
        config_file=make_config(radiance),
        weather_files_path=[WEATHER],
        reward_function=kwargs.pop("reward_function", daylight_reward),
        number_of_timesteps_per_hour=4,
        enable_radiance=radiance,
        **kwargs,
    )


if __name__ == "__main__":
    env = make_env(HERE / "output" / "quickstart",
                   run_period={"begin_year": 2002, "begin_month": 6, "begin_day": 21,
                               "end_year": 2002, "end_month": 6, "end_day": 21})
    obs, info = env.reset()
    for step in range(10_000):
        hour = step / 4
        action = np.array([0.9 if 11 <= hour < 15 else 0.1], dtype=np.float32)  # tint around noon
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 8 == 0 and "raw_next_wpi" in info:
            print(f"{hour:5.2f} h  state={info['raw_next_cfs_state']}  "
                  f"WPI={float(np.ravel(info['raw_next_wpi'])[0]):7.1f} lx  reward={reward:+.3f}")
        if terminated or truncated:
            break
    env.close()
