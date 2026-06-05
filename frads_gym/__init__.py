from gymnasium.envs.registration import register

# 5-Phase-Wiring (gated durch FRADS_USE_FIVE_PHASE) VOR jeder EnergyPlusSetup-
# Instanziierung anwenden, damit der RL-Pfad den 5PM-Zustand der FTG-Kalibrierung
# reproduziert. Ohne das Env-Var = Default 3phase (no-op). Greift in Main-Prozess
# (matrices_manager) und Subprozessen (frads_wrapper), da beide frads_gym importieren.
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
