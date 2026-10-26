"""Roundtrip tests for the per-city normalizer-state pickle mechanism.

These tests exercise save_normalizer_state / load_normalizer_state at the
state-dict level without instantiating a full EnergyPlus simulation. They
cover:
  - per-city pickle naming
  - cold-start behaviour when the pickle file is missing (zero-shot case)
  - error on corrupt/missing version
"""
import os
import pickle
import sys
import pytest


from frads_gym.envs.frads_gym import FradsEnv


def _make_stub_env():
    """Create a FradsEnv-like object with only the normalizer bits populated.

    We avoid calling __init__ because it would try to spin up EnergyPlus.
    Instead we instantiate the class with __new__ and set just the
    attributes exercised by save/load_normalizer_state.
    """
    env = FradsEnv.__new__(FradsEnv)
    env._ema_trackers = {"key_a": {"mean": 1.0, "std": 0.5, "count": 100}}
    env.stat_trackers = {"key_b": {"mean": 2.0, "std": 0.2, "count": 50}}
    env.observation_history = {"key_c": [1, 2, 3]}
    env.observation_history_dynamic = {"key_d": [4, 5, 6]}
    env._obs_welford_trackers = {"key_e": {"count": 10, "mean": 0.0, "M2": 0.0}}
    return env


class TestSaveLoadRoundtrip:
    def test_save_load_preserves_all_trackers(self, tmp_path):
        src = _make_stub_env()
        pickle_path = tmp_path / "normalizer_state_chicago.pkl"
        src.save_normalizer_state(str(pickle_path))
        assert pickle_path.exists()

        dst = FradsEnv.__new__(FradsEnv)
        dst.load_normalizer_state(str(pickle_path))

        assert dst._ema_trackers == src._ema_trackers
        assert dst.stat_trackers == src.stat_trackers
        assert dst.observation_history == src.observation_history
        assert dst.observation_history_dynamic == src.observation_history_dynamic
        assert dst._obs_welford_trackers == src._obs_welford_trackers

    def test_load_missing_file_raises(self, tmp_path):
        dst = FradsEnv.__new__(FradsEnv)
        with pytest.raises(FileNotFoundError):
            dst.load_normalizer_state(str(tmp_path / "does_not_exist.pkl"))

    def test_load_corrupt_version_raises(self, tmp_path):
        pickle_path = tmp_path / "bad.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump({"version": 999, "ema_trackers": {}}, f)

        dst = FradsEnv.__new__(FradsEnv)
        with pytest.raises(ValueError, match="Incompatible normalizer-state version"):
            dst.load_normalizer_state(str(pickle_path))


class TestPerCityDirLookup:
    """Test the directory-based lookup logic used by the new __init__."""

    def test_city_pickle_is_found_in_dir(self, tmp_path):
        """If normalizer_state_<city>.pkl exists in dir, load_from_dir uses it."""
        src = _make_stub_env()
        src.save_normalizer_state(str(tmp_path / "normalizer_state_chicago.pkl"))

        dst = FradsEnv.__new__(FradsEnv)
        loaded = dst._try_load_normalizer_from_dir(str(tmp_path), "chicago")
        assert loaded is True
        assert dst._ema_trackers == src._ema_trackers

    def test_missing_city_pickle_returns_false(self, tmp_path):
        """Houston case: no pickle → returns False, caller cold-starts."""
        # Put a Chicago pickle in the dir, but ask for Houston
        src = _make_stub_env()
        src.save_normalizer_state(str(tmp_path / "normalizer_state_chicago.pkl"))

        dst = FradsEnv.__new__(FradsEnv)
        loaded = dst._try_load_normalizer_from_dir(str(tmp_path), "houston")
        assert loaded is False
