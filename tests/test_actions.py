"""The action vector maps to windows and glazing states in config order."""

from __future__ import annotations

import numpy as np

from frads_gym.envs.frads_gym import FradsEnv

STATES = {"states": [{"name": "clear", "max_threshold": 0.25}, {"name": "light", "max_threshold": 0.5},
                     {"name": "medium", "max_threshold": 0.75}, {"name": "dark", "max_threshold": 1.0}]}


def _env(config):
    env = FradsEnv.__new__(FradsEnv)
    env.epsetup_config = config
    env.observation_space, env.action_space = env.create_custom_spaces(config)
    return env


def _window(low=0.0, high=1.0):
    return {"actuate_cfs_state": {"key": "w"}, "action_space": {"low": low, "high": high}}


def test_action_dimensions_follow_config_order():
    # names deliberately not in alphabetical order
    env = _env({"facade_state_mapping": STATES, "win_top": _window(0.0, 0.5), "win_middle": _window(),
                "win_bottom": _window(0.5, 1.0)})
    assert env.action_space.low.tolist() == [0.0, 0.0, 0.5]
    assert env.action_space.high.tolist() == [0.5, 1.0, 1.0]
    actuated = env._process_action(np.array([0.1, 0.6, 0.9], dtype=np.float32))
    assert list(actuated.items()) == [("win_top", "clear"), ("win_middle", "medium"), ("win_bottom", "dark")]


def test_thresholds_are_inclusive_and_values_clipped():
    env = _env({"facade_state_mapping": STATES, "w": _window()})
    assert [env._map_value_to_facade_state(v) for v in (-1.0, 0.25, 0.2501, 0.75, 2.0)] == \
        ["clear", "clear", "light", "medium", "dark"]
