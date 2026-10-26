"""Tests for ActiveHoursWrapper using a mock Gymnasium environment."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from frads_gym.wrappers.active_hours import ActiveHoursWrapper


class MockFradsEnv(gym.Env):
    """Minimal mock that simulates a day/night cycle over 24 steps (1 step = 1 hour)."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.step_count = 0
        self.total_steps = 72  # 3 days
        self.info = {}

    def reset(self, **kwargs):
        self.step_count = 0
        obs = np.zeros(2, dtype=np.float32)
        self.info = {
            "raw_next_direct_normal_irradiance": 0.0,
            "raw_next_diffuse_horizontal_irradiance": 0.0,
            "raw_next_occupant_count_1": 0.0,
        }
        return obs, self.info

    def step(self, action):
        self.step_count += 1
        hour = self.step_count % 24

        has_solar = 6 <= hour <= 18
        dni = 300.0 if has_solar else 0.0
        dhi = 100.0 if has_solar else 0.0

        has_occ = 8 <= hour <= 17
        occ = 10.0 if has_occ else 0.0

        self.info = {
            "raw_next_direct_normal_irradiance": dni,
            "raw_next_diffuse_horizontal_irradiance": dhi,
            "raw_next_occupant_count_1": occ,
            "agent_active": self.info.get("agent_active", True),
        }

        obs = np.array([float(has_solar), float(has_occ)], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = self.step_count >= self.total_steps

        return obs, reward, terminated, truncated, self.info


class TestSolarOnly:
    """Solar-only gating (default mode=or, only solar enabled)."""

    def test_reset_skips_to_first_solar_hour(self):
        env = ActiveHoursWrapper(MockFradsEnv(), check_solar=True, check_occupancy=False)
        obs, info = env.reset()
        assert obs[0] == 1.0  # has_solar = True

    def test_step_count_matches_solar_hours(self):
        env = ActiveHoursWrapper(MockFradsEnv(), check_solar=True, check_occupancy=False)
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.5]))
            steps += 1
            done = term or trunc
        # 3 days x 13 solar hours (6-18) = 39
        assert steps == 39, f"Expected 39 solar steps, got {steps}"

    def test_agent_active_after_night(self):
        """After a night skip, agent_active must be True on the returned step."""
        env = ActiveHoursWrapper(MockFradsEnv(), check_solar=True, check_occupancy=False)
        env.reset()
        # Run until we hit a night→day transition (agent sees only active steps)
        for _ in range(50):
            _, _, term, trunc, info = env.step(np.array([0.5]))
            assert info.get("agent_active", False), "Wrapper must return only active steps"
            if term or trunc:
                break


class TestOccupancyOnly:
    """Occupancy-only gating."""

    def test_step_count_matches_occupied_hours(self):
        env = ActiveHoursWrapper(MockFradsEnv(), check_solar=False, check_occupancy=True)
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.5]))
            steps += 1
            done = term or trunc
        # 3 days x 10 occupied hours (8-17) = 30
        assert steps == 30, f"Expected 30 occupancy steps, got {steps}"


class TestNoWrapper:
    """Baseline: without wrapper all steps exposed."""

    def test_all_72_steps(self):
        env = MockFradsEnv()
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.5]))
            steps += 1
            done = term or trunc
        assert steps == 72


class TestOrVsAndMode:
    """OR mode (union) should yield more steps than AND mode (intersection)."""

    def test_or_more_than_and(self):
        env_or = ActiveHoursWrapper(
            MockFradsEnv(), check_solar=True, check_occupancy=True, mode="or"
        )
        env_and = ActiveHoursWrapper(
            MockFradsEnv(), check_solar=True, check_occupancy=True, mode="and"
        )

        env_or.reset()
        steps_or = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env_or.step(np.array([0.5]))
            steps_or += 1
            done = term or trunc

        env_and.reset()
        steps_and = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env_and.step(np.array([0.5]))
            steps_and += 1
            done = term or trunc

        # OR: solar(6-18) union occ(8-17) = 6-18 = 13h → 39
        # AND: solar(6-18) intersect occ(8-17) = 8-17 = 10h → 30
        assert steps_or == 39, f"OR mode: expected 39, got {steps_or}"
        assert steps_and == 30, f"AND mode: expected 30, got {steps_and}"
        assert steps_or > steps_and


class TestNoChecksEnabled:
    """Both checks disabled → all steps active."""

    def test_all_steps_when_disabled(self):
        env = ActiveHoursWrapper(MockFradsEnv(), check_solar=False, check_occupancy=False)
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(np.array([0.5]))
            steps += 1
            done = term or trunc
        assert steps == 72
