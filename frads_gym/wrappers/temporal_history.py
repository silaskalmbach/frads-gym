"""Temporal history wrapper — stacks past observations for transformer policies."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np


def compute_flat_dim_map(
    observation_space: gym.spaces.Dict,
    exclude_keys: tuple[str, ...] = ("obs_history", "temporal_mask"),
) -> Tuple[List[Tuple[str, int, int]], int]:
    """Compute (key, start, length) mapping for flattening a Dict obs space.
    Keys are sorted alphabetically (same order as SensorTokenizer).
    Returns:
        flat_dim_map: List of (key, start_index, length)
        flat_obs_dim: Total flattened dimension
    """
    flat_dim_map: List[Tuple[str, int, int]] = []
    offset = 0
    for key in sorted(observation_space.spaces.keys()):
        if key in exclude_keys:
            continue
        space = observation_space.spaces[key]
        length = int(np.prod(space.shape))
        flat_dim_map.append((key, offset, length))
        offset += length
    return flat_dim_map, offset


class TemporalHistoryWrapper(gym.ObservationWrapper):
    """Stack the last *max_time_steps* observations into the Dict obs space.

    Adds two keys:
        obs_history:   (max_time_steps, flat_obs_dim)  float32
        temporal_mask: (max_time_steps,)               float32, 1=valid 0=pad
    """

    def __init__(
        self,
        env: gym.Env,
        max_time_steps: int = 48,
        flatten_keys: Optional[List[str]] = None,
        padding: str = "zero",
    ):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict), (
            "TemporalHistoryWrapper requires a Dict observation space"
        )
        self._max_time_steps = max_time_steps
        self._padding = padding

        # Compute flat_dim_map from the base env's obs space
        self._flat_dim_map, self._flat_obs_dim = compute_flat_dim_map(
            env.observation_space
        )

        # If flatten_keys provided, filter the map
        if flatten_keys is not None:
            self._flat_dim_map = [
                (k, s, l) for k, s, l in self._flat_dim_map if k in flatten_keys
            ]
            recomputed = []
            offset = 0
            for k, _, l in self._flat_dim_map:
                recomputed.append((k, offset, l))
                offset += l
            self._flat_dim_map = recomputed
            self._flat_obs_dim = offset

        # FIFO buffer
        self._history: deque = deque(maxlen=max_time_steps)

        # Extend observation space
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["obs_history"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_time_steps, self._flat_obs_dim),
            dtype=np.float32,
        )
        new_spaces["temporal_mask"] = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(max_time_steps,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        flat = np.zeros(self._flat_obs_dim, dtype=np.float32)
        for key, start, length in self._flat_dim_map:
            flat[start:start + length] = np.asarray(obs[key], dtype=np.float32).ravel()
        return flat

    def _build_history_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        T = self._max_time_steps
        obs_history = np.zeros((T, self._flat_obs_dim), dtype=np.float32)
        temporal_mask = np.zeros(T, dtype=np.float32)
        n_valid = len(self._history)
        offset = T - n_valid
        for i, flat_obs in enumerate(self._history):
            obs_history[offset + i] = flat_obs
            temporal_mask[offset + i] = 1.0
        return obs_history, temporal_mask

    def observation(self, observation):
        self._history.append(self._flatten_obs(observation))
        obs_history, temporal_mask = self._build_history_obs()
        return {**observation, "obs_history": obs_history, "temporal_mask": temporal_mask}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history.clear()
        self._history.append(self._flatten_obs(obs))
        obs_history, temporal_mask = self._build_history_obs()
        return {**obs, "obs_history": obs_history, "temporal_mask": temporal_mask}, info

    @property
    def flat_dim_map(self) -> List[Tuple[str, int, int]]:
        return list(self._flat_dim_map)
