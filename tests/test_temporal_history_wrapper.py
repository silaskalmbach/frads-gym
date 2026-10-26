"""Tests for TemporalHistoryWrapper."""

from __future__ import annotations

import sys
import os

import gymnasium as gym
import numpy as np
import pytest


from frads_gym.wrappers.temporal_history import TemporalHistoryWrapper, compute_flat_dim_map


def make_dict_env(n_sensors: int = 5) -> gym.Env:
    spaces_dict = {
        "datetime": gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
    }
    sensor_keys = [
        "avg_wpi_1",
        "cooling_energy_1",
        "cfs_state_top",
        "direct_normal_irradiance",
        "indoor_temp_1",
    ]
    for key in sensor_keys[:n_sensors]:
        spaces_dict[key] = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    obs_space = gym.spaces.Dict(spaces_dict)
    act_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    class _Env(gym.Env):
        observation_space = obs_space
        action_space = act_space
        _step = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            return self._obs(), {}

        def step(self, action):
            self._step += 1
            return self._obs(), 0.0, self._step >= 50, False, {}

        def _obs(self):
            return {k: space.sample() for k, space in self.observation_space.spaces.items()}

    return _Env()


class TestComputeFlatDimMap:
    def test_sorted_keys(self):
        env = make_dict_env(n_sensors=3)
        fdm, flat_dim = compute_flat_dim_map(env.observation_space)
        keys = [k for k, _, _ in fdm]
        assert keys == sorted(keys)

    def test_total_dim(self):
        env = make_dict_env(n_sensors=5)
        fdm, flat_dim = compute_flat_dim_map(env.observation_space)
        assert flat_dim == 9  # datetime(4) + 5 sensors(1 each)

    def test_offsets_contiguous(self):
        env = make_dict_env(n_sensors=5)
        fdm, flat_dim = compute_flat_dim_map(env.observation_space)
        for i, (key, start, length) in enumerate(fdm):
            if i > 0:
                prev_key, prev_start, prev_length = fdm[i - 1]
                assert start == prev_start + prev_length
        last_key, last_start, last_length = fdm[-1]
        assert last_start + last_length == flat_dim

    def test_excludes_history_keys(self):
        env = make_dict_env(n_sensors=3)
        wrapped_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            "obs_history": gym.spaces.Box(low=-1, high=1, shape=(8, 7)),
            "temporal_mask": gym.spaces.Box(low=0, high=1, shape=(8,)),
        })
        fdm, flat_dim = compute_flat_dim_map(wrapped_space)
        keys = [k for k, _, _ in fdm]
        assert "obs_history" not in keys
        assert "temporal_mask" not in keys


class TestTemporalHistoryWrapper:
    def test_observation_space_extension(self):
        env = make_dict_env(n_sensors=5)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=8)
        assert "obs_history" in wrapped.observation_space.spaces
        assert "temporal_mask" in wrapped.observation_space.spaces
        assert wrapped.observation_space["obs_history"].shape == (8, 9)
        assert wrapped.observation_space["temporal_mask"].shape == (8,)

    def test_original_keys_preserved(self):
        env = make_dict_env(n_sensors=5)
        orig_keys = set(env.observation_space.spaces.keys())
        wrapped = TemporalHistoryWrapper(env, max_time_steps=8)
        assert orig_keys.issubset(wrapped.observation_space.spaces.keys())

    def test_reset_returns_valid_obs(self):
        env = make_dict_env(n_sensors=3)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=4)
        obs, info = wrapped.reset()
        assert obs["obs_history"].shape == (4, 7)
        assert obs["temporal_mask"].shape == (4,)

    def test_mask_after_reset(self):
        env = make_dict_env(n_sensors=3)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=4)
        obs, _ = wrapped.reset()
        mask = obs["temporal_mask"]
        np.testing.assert_array_equal(mask[:3], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[3:], [1.0])

    def test_mask_progression(self):
        env = make_dict_env(n_sensors=3)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=4)
        obs, _ = wrapped.reset()
        np.testing.assert_array_equal(obs["temporal_mask"], [0, 0, 0, 1])
        obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
        np.testing.assert_array_equal(obs["temporal_mask"], [0, 0, 1, 1])
        obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
        np.testing.assert_array_equal(obs["temporal_mask"], [0, 1, 1, 1])
        obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
        np.testing.assert_array_equal(obs["temporal_mask"], [1, 1, 1, 1])

    def test_fifo_behavior(self):
        env = make_dict_env(n_sensors=3)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=3)
        obs, _ = wrapped.reset()
        first_row = obs["obs_history"][-1].copy()
        for _ in range(3):
            obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
        assert not np.allclose(obs["obs_history"][0], first_row)
        np.testing.assert_array_equal(obs["temporal_mask"], [1, 1, 1])

    def test_reset_clears_history(self):
        env = make_dict_env(n_sensors=3)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=4)
        wrapped.reset()
        for _ in range(5):
            wrapped.step(wrapped.action_space.sample())
        obs, _ = wrapped.reset()
        np.testing.assert_array_equal(obs["temporal_mask"], [0, 0, 0, 1])

    def test_different_max_time_steps(self):
        env = make_dict_env(n_sensors=3)
        for T in [4, 8, 24, 48]:
            wrapped = TemporalHistoryWrapper(env, max_time_steps=T)
            obs, _ = wrapped.reset()
            assert obs["obs_history"].shape == (T, 7)
            assert obs["temporal_mask"].shape == (T,)

    def test_flat_dim_map_attribute(self):
        env = make_dict_env(n_sensors=3)
        wrapped = TemporalHistoryWrapper(env, max_time_steps=4)
        fdm = wrapped.flat_dim_map
        assert isinstance(fdm, list)
        assert len(fdm) > 0
        assert all(len(t) == 3 for t in fdm)
