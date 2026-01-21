from gymnasium.envs.registration import register

register(
    id="frads-gym/FradsEnv-v0",
    entry_point="frads_gym.envs:FradsEnv",
)
