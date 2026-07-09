"""Action repeat wrapper — holds the agent's action for N consecutive sim steps.

Densifies per-decision credit for a fixed decision cadence: the agent
observes and acts once every ``repeat`` simulation steps; the chosen action
is held constant for the intervening steps.  Rewards from all ``repeat``
inner steps are summed and returned as the single decision-step reward.

Every inner ``env.step()`` call still runs through the base env (FradsEnv),
so ``environment_log.csv`` keeps one row PER SIM STEP, not per decision —
the action column repeats across a block, but reward/obs/normalizer-tracker
rows stay at the original cadence (e.g. 15-min). This preserves gate-eval
frame compatibility (evaluation code that reads the log at sim-step
granularity, e.g. hvaccr_gate2, is unaffected).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat the agent's action for *repeat* consecutive base-env steps.

    Parameters:
        env: Base environment (or an already-wrapped env) exposing the
            standard Gymnasium step/reset API.
        repeat: Number of consecutive base-env steps per decision.
            ``repeat=1`` is a no-op (one base-env step per decision).
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        assert repeat >= 1, f"repeat must be >= 1, got {repeat}"
        self.repeat = repeat

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        total_reward = 0.0
        obs, reward, terminated, truncated, info = None, 0.0, False, False, {}
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
