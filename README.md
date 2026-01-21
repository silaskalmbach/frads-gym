# frads-gym

A Gymnasium environment for building facade control with [frads](https://github.com/LBNL-ETA/frads) (framework for radiance simulation).

## Features

- **Building Simulation**: Integrates EnergyPlus for thermal simulation
- **Facade Control**: Control electrochromic glazing states
- **Gymnasium Compatible**: Full compatibility with stable-baselines3 and other RL frameworks

## Installation

### From Source (Development)

```bash
git clone https://github.com/silaskalmbach/frads-gym.git
cd frads-gym
pip install -e .
```

### As Submodule in Parent Project

```bash
pip install -e ./gymnasium
```

## Usage

```python
import gymnasium
import frads_gym

# Create the environment
env = gymnasium.make('frads-gym/FradsEnv-v0', config_file='path/to/config.json')

# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Configuration

The environment requires a JSON configuration file specifying:
- EnergyPlus model path
- Weather file path
- Observation and action space definitions
- Facade state mappings

See the `configs/` directory for examples.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Silas Kalmbach (silas.kalmbach@gmail.com)
