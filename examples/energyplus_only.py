"""EnergyPlus without Radiance: the glazing state still changes the thermal behaviour.

Runs the same summer day twice, once clear and once fully tinted, and compares the sensible
cooling energy of the zone. Without Radiance no daylight matrices are needed, so this is fast.

    python energyplus_only.py
"""

import numpy as np

from frads_gym import FradsEnv
from quickstart import HERE, WEATHER, ZONE, make_config

config = make_config(radiance=False)
config["cooling"] = {"get_variable_value": {"name": "Zone Air System Sensible Cooling Energy", "key": ZONE},
                     "observation_space": {"low": 0, "high": 1e9}}

env = FradsEnv(output_dir=str(HERE / "output" / "energyplus_only"), config_file=config,
               weather_files_path=[WEATHER], number_of_timesteps_per_hour=4, enable_radiance=False,
               reward_function=lambda obs, action, next_obs, info: 0.0,
               run_period={"begin_year": 2002, "begin_month": 6, "begin_day": 21,
                           "end_year": 2002, "end_month": 6, "end_day": 21})

for label, action in (("clear", 0.1), ("dark", 0.9)):
    env.reset()
    cooling_j = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        if terminated or truncated:
            break
        cooling_j += float(np.ravel(info["raw_next_cooling"])[0])
    print(f"{label:5s}: sensible cooling {cooling_j / 3.6e6:6.1f} kWh")
env.close()
