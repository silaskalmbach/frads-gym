<p align="center">
  <img src="https://raw.githubusercontent.com/silaskalmbach/frads-gym/main/docs/assets/logo.png" alt="frads-gym logo" width="200"/>
</p>

# frads-gym

[![CI](https://github.com/silaskalmbach/frads-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/silaskalmbach/frads-gym/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18504304.svg)](https://doi.org/10.5281/zenodo.18504304)

`frads-gym` is a [Gymnasium](https://gymnasium.farama.org/) environment for controlling switchable
facades, such as electrochromic glazing, with reinforcement learning. It couples
[EnergyPlus](https://energyplus.net/) (thermal and energy simulation) and
[Radiance](https://www.radiance-online.org/) (daylight and glare) through
[frads](https://github.com/LBNL-ETA/frads) and hands control to the agent at every EnergyPlus timestep.

- One `step()` per EnergyPlus zone timestep; the action sets the glazing state of each controlled window.
- Observations from a JSON configuration: any EnergyPlus output variable, work-plane or sensor
  illuminance, enhanced simplified DGP, daylight-dimmed lighting energy, energy sums and more.
- Per-observation normalisation (static, dynamic percentile, z-score, EMA), episodes per day or month,
  wrappers for action repeat, active hours, solar forecast and observation history.
- `pip install` brings EnergyPlus and Radiance along; nothing has to be installed by hand.

## Installation

```bash
pip install frads-gym
```

Python 3.10 to 3.12 on Linux (x86_64), Windows (x86_64) and macOS (arm64). The five-phase method and
image-based DGP need the frads fork: `pip install -r requirements-5pm.txt`
(see [Installation](https://silaskalmbach.github.io/frads-gym/installation/)).

## Quickstart

```bash
git clone https://github.com/silaskalmbach/frads-gym.git
cd frads-gym/examples
python quickstart.py
```

`quickstart.py` simulates one June day in the DOE medium office with Chicago weather (both ship with
`pyenergyplus`) and four synthetic electrochromic states created by `make_glazing.py`. The core of it:

```python
from frads_gym import FradsEnv

env = FradsEnv(output_dir="output", config_file=config, weather_files_path=[epw],
               reward_function=lambda obs, action, next_obs, info: 0.0,
               number_of_timesteps_per_hour=4)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Documentation

https://silaskalmbach.github.io/frads-gym/: installation, how the co-simulation works, the full
configuration reference, wrappers and the API.

## Citation

Please cite the archived release you used; the concept DOI
[10.5281/zenodo.18504304](https://doi.org/10.5281/zenodo.18504304) always resolves to the latest
version. `CITATION.cff` holds the metadata.

## License

MIT, see [LICENSE](https://github.com/silaskalmbach/frads-gym/blob/main/LICENSE).
