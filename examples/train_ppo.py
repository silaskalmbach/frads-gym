"""Train a PPO agent for a few simulated days (needs: pip install stable-baselines3).

This only shows the wiring. A useful policy needs far more training, a reward that weighs
energy, glare and daylight, and usually several environments in parallel (SubprocVecEnv:
frads-gym runs one simulation per process).

    python train_ppo.py
"""

from stable_baselines3 import PPO

from quickstart import HERE, make_env

env = make_env(HERE / "output" / "train_ppo",
               run_period={"begin_year": 2002, "begin_month": 6, "begin_day": 17,
                           "end_year": 2002, "end_month": 6, "end_day": 23})
model = PPO("MultiInputPolicy", env, n_steps=96, batch_size=48, verbose=1)
model.learn(total_timesteps=96 * 3)  # three days at 15-min steps
model.save(HERE / "output" / "train_ppo" / "ppo_facade")
env.close()
