"""Tests for ActionRepeatWrapper."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from frads_gym.wrappers.action_repeat import ActionRepeatWrapper


class _CountingEnv(gym.Env):
    """Deterministic env: reward = 1.0 per step, obs = step counter.

    Optionally terminates/truncates at a fixed inner-step count, so tests
    can verify early-stop behaviour inside a repeat block.
    """

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, terminate_at: int = None, truncate_at: int = None):
        self._step_count = 0
        self._terminate_at = terminate_at
        self._truncate_at = truncate_at
        self.actions_seen = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.actions_seen = []
        return self._obs(), {"reset": True}

    def step(self, action):
        self._step_count += 1
        self.actions_seen.append(np.asarray(action).copy())
        terminated = self._terminate_at is not None and self._step_count >= self._terminate_at
        truncated = self._truncate_at is not None and self._step_count >= self._truncate_at
        return self._obs(), 1.0, terminated, truncated, {"step_count": self._step_count}

    def _obs(self):
        return np.array([self._step_count], dtype=np.float32)


class TestActionRepeatWrapper:
    def test_repeat_one_is_noop(self):
        env = ActionRepeatWrapper(_CountingEnv(), repeat=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert reward == 1.0
        assert info["step_count"] == 1
        assert not terminated and not truncated

    def test_reward_summing(self):
        env = ActionRepeatWrapper(_CountingEnv(), repeat=8)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert reward == 8.0
        assert info["step_count"] == 8
        assert not terminated and not truncated

    def test_action_held_constant_across_block(self):
        inner = _CountingEnv()
        env = ActionRepeatWrapper(inner, repeat=8)
        env.reset()
        a = np.array([0.7], dtype=np.float32)
        env.step(a)
        assert len(inner.actions_seen) == 8
        for seen in inner.actions_seen:
            np.testing.assert_array_equal(seen, a)

    def test_early_truncation_returns_accumulated_reward(self):
        env = ActionRepeatWrapper(_CountingEnv(truncate_at=3), repeat=8)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert reward == 3.0  # only 3 inner steps ran before truncation
        assert truncated is True
        assert terminated is False
        assert info["step_count"] == 3

    def test_early_termination_returns_accumulated_reward(self):
        env = ActionRepeatWrapper(_CountingEnv(terminate_at=5), repeat=8)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert reward == 5.0
        assert terminated is True
        assert info["step_count"] == 5

    def test_obs_passthrough_is_last_inner_obs(self):
        env = ActionRepeatWrapper(_CountingEnv(), repeat=4)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert obs[0] == 4  # last inner step's observation (step_count=4)

    def test_reset_forwards_to_base_env(self):
        env = ActionRepeatWrapper(_CountingEnv(), repeat=8)
        obs, info = env.reset()
        assert info == {"reset": True}
        assert obs[0] == 0

    def test_invalid_repeat_raises(self):
        with pytest.raises(AssertionError):
            ActionRepeatWrapper(_CountingEnv(), repeat=0)

    def test_multiple_decision_steps_accumulate_inner_steps(self):
        env = ActionRepeatWrapper(_CountingEnv(), repeat=8)
        env.reset()
        obs1, r1, t1, tr1, info1 = env.step(np.array([0.5], dtype=np.float32))
        assert info1["step_count"] == 8
        obs2, r2, t2, tr2, info2 = env.step(np.array([0.2], dtype=np.float32))
        assert info2["step_count"] == 16
        assert r2 == 8.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
