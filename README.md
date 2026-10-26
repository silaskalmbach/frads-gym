<p align="center">
  <img src="https://raw.githubusercontent.com/silaskalmbach/frads-gym/main/docs/assets/logo.png" alt="frads-gym logo" width="200"/>
</p>

# frads-gym

[![CI](https://github.com/silaskalmbach/frads-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/silaskalmbach/frads-gym/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18504304.svg)](https://doi.org/10.5281/zenodo.18504304)
[![Docs](https://img.shields.io/badge/docs-silaskalmbach.github.io-blue)](https://silaskalmbach.github.io/frads-gym/)

**Documentation: <https://silaskalmbach.github.io/frads-gym/>**

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

Python 3.10 to 3.12 on Linux (x86_64), Windows (x86_64) and macOS (arm64). Details and the development
install: [Installation](https://silaskalmbach.github.io/frads-gym/installation/).

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

# config: path to a JSON configuration or a dict (quickstart.py builds one); epw: path to a weather file
env = FradsEnv(output_dir="output", config_file=config, weather_files_path=[epw],
               reward_function=lambda obs, action, next_obs, info: 0.0,
               number_of_timesteps_per_hour=4)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The other examples (EnergyPlus without Radiance, the wiring for PPO with stable-baselines3) are listed
in [Quickstart](https://silaskalmbach.github.io/frads-gym/quickstart/).

## Documentation

<https://silaskalmbach.github.io/frads-gym/>

- [How it works](https://silaskalmbach.github.io/frads-gym/how-it-works/): the co-simulation loop, timing of
  actions and observations, episodes, reproducibility
- [Configuration](https://silaskalmbach.github.io/frads-gym/configuration/): every key of the JSON configuration
- [Wrappers](https://silaskalmbach.github.io/frads-gym/wrappers/) and
  [API reference](https://silaskalmbach.github.io/frads-gym/api/)

## Contributing

Bug reports, questions and pull requests are welcome, see
[CONTRIBUTING.md](https://github.com/silaskalmbach/frads-gym/blob/main/CONTRIBUTING.md).

## Citation

Please cite the archived release you used; the concept DOI
[10.5281/zenodo.18504304](https://doi.org/10.5281/zenodo.18504304) always resolves to the latest
version. `CITATION.cff` holds the metadata.

## License

MIT, see [LICENSE](https://github.com/silaskalmbach/frads-gym/blob/main/LICENSE).
