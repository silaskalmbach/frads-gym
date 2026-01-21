from gymnasium.envs.registration import register
from frads_gym.envs.frads_gym import FradsEnv

# Register the environment with gymnasium
register(
    id="frads-gym/FradsEnv-v0",
    entry_point="frads_gym.envs:FradsEnv",
)

# Make FradsEnv available at package level
__all__ = ["FradsEnv"]
