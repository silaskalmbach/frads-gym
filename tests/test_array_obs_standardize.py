"""Tests for observation_standardize support in the ARRAY path of
FradsEnv._process_observation.

Bug (verified 2026-06-09): the array path only knew observation_normalize.
With ema_zscore mode (normalize inactive, standardize active — as set by
rl/reward_candidates/_reward_settings.py::_apply_obs_normalization_mode),
array observations (e.g. Radiance WPI shape-(1,) arrays, edgps [DGP, Ev])
passed through RAW and unclipped.

The fix standardizes array obs element-wise with a per-element tracker key
"key[i]" (elements of one array obs have different scales and must not share
EMA statistics), while passing the parent key's config explicitly.

All tests use the bare-object pattern (FradsEnv.__new__) — no EnergyPlus.
"""
import os
import sys

import numpy as np
import pytest
from gymnasium import spaces


from frads_gym.envs.frads_gym import FradsEnv


def _make_env(epsetup_config, obs_spaces):
    """Bare FradsEnv with only the attributes _process_observation needs."""
    env = FradsEnv.__new__(FradsEnv)
    env.epsetup_config = epsetup_config
    env.observation_space = spaces.Dict(obs_spaces)
    return env


def _array_standardize_env(ema_span=100):
    """Array obs (2,) with ema_zscore-style config: standardize active,
    normalize inactive. Element scales differ deliberately (DGP ~0.3 vs
    Ev ~5000 lux). observation_space bounds are per-element lists."""
    config = {
        "edgps_1": {
            "observation_space": {"low": [-1.0, -1.0], "high": [1.0, 1.0]},
            "observation_standardize": {
                "active": True,
                "dynamic": "ema",
                "ema_span": ema_span,
                "low": -3,
                "high": 3,
            },
            "observation_normalize": {
                "active": False,
                "low": [0.0, 0.0],
                "high": [1.0, 10000.0],
            },
        }
    }
    obs_spaces = {
        "edgps_1": spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]),
            shape=(2,), dtype=np.float32,
        )
    }
    return _make_env(config, obs_spaces)


class TestArrayStandardize:
    """(a) Array obs with active observation_standardize (ema)."""

    def test_output_in_bounds_and_not_raw(self):
        env = _array_standardize_env()
        raw = [0.3, 5000.0]
        out = env._process_observation({"edgps_1": raw})["edgps_1"]

        assert out.shape == (2,)
        assert out.dtype == np.float32
        assert np.all(out >= -1.0) and np.all(out <= 1.0)
        # 5000 lux must NOT pass through raw anymore
        assert not np.allclose(out, np.array(raw, dtype=np.float32))

    def test_first_call_maps_to_midpoint(self):
        # Warmup, count=1: mean=value, std=1 -> z=0 -> midpoint of [-1, 1]
        env = _array_standardize_env()
        out = env._process_observation({"edgps_1": [0.3, 5000.0]})["edgps_1"]
        assert np.array_equal(out, np.array([0.0, 0.0], dtype=np.float32))

    def test_per_element_trackers_with_distinct_means(self):
        env = _array_standardize_env()
        for i in range(10):
            out = env._process_observation(
                {"edgps_1": [0.3 + 0.01 * i, 5000.0 + 50.0 * i]}
            )["edgps_1"]
            assert np.all(out >= -1.0) and np.all(out <= 1.0)

        assert "edgps_1[0]" in env._ema_trackers
        assert "edgps_1[1]" in env._ema_trackers
        mean0 = env._ema_trackers["edgps_1[0]"]["mean"]
        mean1 = env._ema_trackers["edgps_1[1]"]["mean"]
        # Elements must NOT share one tracker: means stay on their own scale
        assert 0.0 < mean0 < 1.0
        assert mean1 > 4000.0
        assert abs(mean1 - mean0) > 100.0
        # No tracker under the parent key itself
        assert "edgps_1" not in env._ema_trackers


class TestArrayNormalizeRegression:
    """(b) Array obs with active observation_normalize: bitwise old behavior."""

    def test_static_minmax_exact(self):
        config = {
            "wpi_arr": {
                "observation_space": {"low": [0.0, 0.0], "high": [1.0, 1.0]},
                "observation_normalize": {
                    "active": True,
                    "low": [0.0, 0.0],
                    "high": [1.0, 10000.0],
                },
                "observation_standardize": {
                    "active": False, "dynamic": "ema", "low": -3, "high": 3,
                },
            }
        }
        obs_spaces = {
            "wpi_arr": spaces.Box(
                low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]),
                shape=(2,), dtype=np.float32,
            )
        }
        env = _make_env(config, obs_spaces)
        out = env._process_observation({"wpi_arr": [0.3, 5000.0]})["wpi_arr"]

        # Hand-computed static min-max: (0.3-0)/(1-0)=0.3, (5000-0)/(10000-0)=0.5
        expected = np.array([0.3, 0.5], dtype=np.float32)
        assert np.array_equal(out, expected)
        # Standardization must not have run: no EMA trackers created
        assert getattr(env, "_ema_trackers", {}) == {}


class TestArrayRawPassthroughRegression:
    """(c) Array obs with neither normalize nor standardize active:
    raw passthrough, unclipped (training-true for existing models)."""

    def test_raw_passthrough_unchanged(self):
        config = {
            "wpi_raw": {
                "observation_space": {"low": [0.0, 0.0], "high": [1.0, 1.0]},
                "observation_normalize": {
                    "active": False, "low": [0.0, 0.0], "high": [1.0, 10000.0],
                },
                "observation_standardize": {
                    "active": False, "dynamic": "ema", "low": -3, "high": 3,
                },
            }
        }
        obs_spaces = {
            "wpi_raw": spaces.Box(
                low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]),
                shape=(2,), dtype=np.float32,
            )
        }
        env = _make_env(config, obs_spaces)
        raw = [123.4, 5678.9]
        out = env._process_observation({"wpi_raw": raw})["wpi_raw"]

        # Exactly the raw values as float32 — no clipping, no scaling
        assert np.array_equal(out, np.array(raw, dtype=np.float32))
        assert getattr(env, "_ema_trackers", {}) == {}


class TestScalarStandardizeRegression:
    """(d) Scalar obs with active standardize: unchanged behavior."""

    def test_first_call_warmup_midpoint(self):
        config = {
            "wpi_scalar": {
                "observation_space": {"low": -1.0, "high": 1.0},
                "observation_standardize": {
                    "active": True,
                    "dynamic": "ema",
                    "ema_span": 100,
                    "low": -3,
                    "high": 3,
                },
            }
        }
        obs_spaces = {
            "wpi_scalar": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        }
        env = _make_env(config, obs_spaces)
        out = env._process_observation({"wpi_scalar": 500.0})["wpi_scalar"]

        # First call: count=1, mean=value, std=1 -> z=0 -> midpoint of [-1,1]
        assert out.shape == (1,)
        assert np.array_equal(out, np.array([0.0], dtype=np.float32))
        # Scalar path keeps the plain key (no "[i]" suffix)
        assert "wpi_scalar" in env._ema_trackers
        assert "wpi_scalar[0]" not in env._ema_trackers


class TestNormalizerStateRoundtrip:
    """(e) Per-element trackers survive save/load_normalizer_state."""

    def test_element_trackers_roundtrip(self, tmp_path):
        env = _array_standardize_env()
        for i in range(5):
            env._process_observation(
                {"edgps_1": [0.3 + 0.01 * i, 5000.0 + 50.0 * i]}
            )
        assert "edgps_1[0]" in env._ema_trackers
        assert "edgps_1[1]" in env._ema_trackers

        pickle_path = tmp_path / "normalizer_state_test.pkl"
        env.save_normalizer_state(str(pickle_path))

        fresh = FradsEnv.__new__(FradsEnv)
        fresh.load_normalizer_state(str(pickle_path))

        assert "edgps_1[0]" in fresh._ema_trackers
        assert "edgps_1[1]" in fresh._ema_trackers
        assert fresh._ema_trackers == env._ema_trackers
