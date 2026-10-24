from gymnasium.envs.registration import register

# Apply the five-phase wiring (switched on by FRADS_USE_FIVE_PHASE) before any EnergyPlusSetup is
# created, in every process that imports frads_gym. Without the variable it changes nothing.
from frads_gym.five_phase_wiring import apply_five_phase_wiring
apply_five_phase_wiring()

from frads_gym.envs.frads_gym import FradsEnv

# Register the environment with gymnasium
register(
    id="frads-gym/FradsEnv-v0",
    entry_point="frads_gym.envs:FradsEnv",
)

# Make FradsEnv available at package level
__all__ = ["FradsEnv"]
