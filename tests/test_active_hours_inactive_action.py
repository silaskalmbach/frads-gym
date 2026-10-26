"""Tests for ActiveHoursWrapper.inactive_action using a mock environment.

Covers the v8 night-tint option:
  - Default (inactive_action=None): legacy behaviour — fast-forwarded
    (inactive) steps replay the agent's LAST action; the reset fast-forward
    before the first agent step uses zeros.
  - inactive_action=0.0: every fast-forwarded step receives the constant
    0.0 on all action dims (FTG: fully tinted), active steps still receive
    the agent's action unchanged.

No EnergyPlus/Radiance involved (bare mock, mirrors
tests/test_active_hours_wrapper.py).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from frads_gym.wrappers.active_hours import ActiveHoursWrapper


class RecordingMockEnv(gym.Env):
    """Day/night mock (1 step = 1 hour, solar 6-18) that records every
    action passed to ``step`` together with whether the call happened while
    the env was inactive (i.e. it must be a wrapper fast-forward call).
    """

    def __init__(self, n_actions: int = 4):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(n_actions,), dtype=np.float32)
        self.step_count = 0
        self.total_steps = 72  # 3 days
        self.info = {}
        # list of (is_fast_forward_call, action_copy)
        self.received = []

    def _next_is_active(self) -> bool:
        return (
            self.info.get("raw_next_direct_normal_irradiance", 0.0)
            + self.info.get("raw_next_diffuse_horizontal_irradiance", 0.0)
        ) > 0

    def reset(self, **kwargs):
        self.step_count = 0
        self.received = []
        self.info = {
            "raw_next_direct_normal_irradiance": 0.0,
            "raw_next_diffuse_horizontal_irradiance": 0.0,
        }
        return np.zeros(2, dtype=np.float32), self.info

    def step(self, action):
        # If the env was inactive BEFORE this call, the wrapper can only be
        # fast-forwarding (it never forwards agent actions while inactive).
        self.received.append((not self._next_is_active(), np.array(action, copy=True)))

        self.step_count += 1
        hour = self.step_count % 24
        has_solar = 6 <= hour <= 18
        self.info = {
            "raw_next_direct_normal_irradiance": 300.0 if has_solar else 0.0,
            "raw_next_diffuse_horizontal_irradiance": 100.0 if has_solar else 0.0,
        }
        obs = np.array([float(has_solar), 0.0], dtype=np.float32)
        truncated = self.step_count >= self.total_steps
        return obs, 1.0, False, truncated, self.info


def _run_episode(wrapper: ActiveHoursWrapper, mock: RecordingMockEnv):
    """Drive a full episode with distinct agent actions per active step.

    Returns (agent_actions, ff_records) where ff_records is the mock's list
    of fast-forwarded (is_ff=True) action copies in call order.
    """
    wrapper.reset()
    agent_actions = []
    done = False
    i = 0
    while not done:
        action = np.full(mock.action_space.shape, 0.1 + 0.01 * i, dtype=np.float32)
        agent_actions.append(action)
        _, _, term, trunc, _ = wrapper.step(action)
        done = term or trunc
        i += 1
    return agent_actions, mock.received


class TestDefaultBehaviour:
    """inactive_action=None (default) must reproduce the legacy behaviour."""

    def test_reset_fast_forward_uses_zeros(self):
        mock = RecordingMockEnv()
        wrapper = ActiveHoursWrapper(mock, check_solar=True, check_occupancy=False)
        wrapper.reset()
        ff = [a for is_ff, a in mock.received if is_ff]
        assert len(ff) > 0, "reset must fast-forward through the initial night"
        for a in ff:
            np.testing.assert_array_equal(a, np.zeros(mock.action_space.shape))

    def test_night_steps_hold_last_agent_action(self):
        mock = RecordingMockEnv()
        wrapper = ActiveHoursWrapper(mock, check_solar=True, check_occupancy=False)
        _run_episode(wrapper, mock)

        last_agent_action = None
        ff_after_agent = 0
        for is_ff, a in mock.received:
            if is_ff:
                if last_agent_action is not None:
                    np.testing.assert_array_equal(a, last_agent_action)
                    ff_after_agent += 1
            else:
                last_agent_action = a
        assert ff_after_agent > 0, "episode must contain night fast-forward steps"

    def test_active_step_count_unchanged(self):
        mock = RecordingMockEnv()
        wrapper = ActiveHoursWrapper(mock, check_solar=True, check_occupancy=False)
        agent_actions, _ = _run_episode(wrapper, mock)
        # 3 days x 13 solar hours (6-18) = 39 (same as test_active_hours_wrapper)
        assert len(agent_actions) == 39


class TestInactiveActionConstant:
    """inactive_action=0.0 (v8): nights fully tinted on all action dims."""

    def test_inactive_steps_get_constant(self):
        mock = RecordingMockEnv()
        wrapper = ActiveHoursWrapper(
            mock, check_solar=True, check_occupancy=False, inactive_action=0.0
        )
        _run_episode(wrapper, mock)
        ff = [a for is_ff, a in mock.received if is_ff]
        assert len(ff) > 0
        for a in ff:
            np.testing.assert_array_equal(a, np.zeros(mock.action_space.shape))

    def test_reset_fast_forward_gets_constant(self):
        mock = RecordingMockEnv()
        wrapper = ActiveHoursWrapper(
            mock, check_solar=True, check_occupancy=False, inactive_action=0.3
        )
        wrapper.reset()
        ff = [a for is_ff, a in mock.received if is_ff]
        assert len(ff) > 0
        for a in ff:
            np.testing.assert_array_equal(
                a, np.full(mock.action_space.shape, 0.3, dtype=np.float32)
            )

    def test_active_steps_keep_agent_action(self):
        mock = RecordingMockEnv()
        wrapper = ActiveHoursWrapper(
            mock, check_solar=True, check_occupancy=False, inactive_action=0.0
        )
        agent_actions, received = _run_episode(wrapper, mock)
        active_received = [a for is_ff, a in received if not is_ff]
        assert len(active_received) == len(agent_actions) == 39
        for sent, got in zip(agent_actions, active_received):
            np.testing.assert_array_equal(sent, got)

    def test_constant_broadcast_to_all_panes(self):
        """4-pane action space: scalar inactive_action covers every pane."""
        mock = RecordingMockEnv(n_actions=4)
        wrapper = ActiveHoursWrapper(
            mock, check_solar=True, check_occupancy=False, inactive_action=0.0
        )
        assert wrapper._inactive_action_arr.shape == (4,)
        _run_episode(wrapper, mock)
        ff = [a for is_ff, a in mock.received if is_ff]
        for a in ff:
            assert a.shape == (4,)
            np.testing.assert_array_equal(a, np.zeros(4, dtype=np.float32))
