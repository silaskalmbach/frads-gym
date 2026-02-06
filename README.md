# frads-gym
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18504305.svg)](https://doi.org/10.5281/zenodo.18504305)
A Gymnasium environment for building facade control with [frads](https://github.com/LBNL-ETA/frads) (framework for radiance simulation).
It combines **EnergyPlus** (thermal simulation) and **Radiance** (lighting simulation) for building facade control with reinforcement learning.

## Features

- **Coupled Simulation**: Combines EnergyPlus thermal simulation with Radiance lighting simulation
- **Facade Control**: Control electrochromic glazing states, enabling smart facade optimization
- **Lighting Metrics**: Capture lighting performance metrics including:
  - **eDGP** (Enhanced Daylight Glare Probability)
  - **WPI** (Workplane Illuminance)
- **Energy Optimization**: Monitor and optimize cooling, heating, and lighting energy
- **Gymnasium Compatible**: Full compatibility with stable-baselines3, RLlib, and other RL frameworks

## Installation

### From Source (Development)

```bash
git clone https://github.com/silaskalmbach/frads-gym.git
cd frads-gym
pip install -e .
```

## Usage

```python
import json
from frads_gym import FradsEnv

# Define a simple reward function
def reward_function(state, action, next_state, info):
    energy = info.get('raw_next_total_energy_demand_1', [0.0])[0]
    return -energy  # Minimize energy consumption

# Load configuration
with open('path/to/config.json', 'r') as f:
    config = json.load(f)
config['config_dir'] = 'path/to/config/directory'

# Create the environment
env = FradsEnv(
    output_dir='./simulation_output',
    config_file=config,
    weather_files_path=['path/to/weather.epw'],
    reward_function=reward_function,
    enable_radiance=True  # Enable lighting simulation
)

# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See `examples/` for a complete example with electrochromic glazing control.

## Configuration

The environment requires a JSON configuration file specifying:
- EnergyPlus model path (`.idf` file)
- Glazing system definitions (for Radiance)
- Weather file path (`.epw` file)
- Observation and action space definitions
- Facade state mappings

See the `examples/data/` directory for example configurations.

## Dependencies

- [frads](https://github.com/LBNL-ETA/frads) - Framework for Radiance Simulation
- [gymnasium](https://gymnasium.farama.org/) - RL Environment API
- [EnergyPlus](https://energyplus.net/) - Building Energy Simulation
- [Radiance](https://www.radiance-online.org/) - Lighting Simulation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author


## Citation

If you use this software in your research, please cite it using the following BibTeX entry:

```bibtex
@software{frads_gym_2026,
  author = {Kalmbach, Silas},
  doi = {10.5281/zenodo.18504305},
  month = {2},
  title = {{frads-gym: A Gymnasium environment for building facade control}},
  url = {https://github.com/silaskalmbach/frads-gym},
  version = {0.1.1},
  year = {2026}
}
```
