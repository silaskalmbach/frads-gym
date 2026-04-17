"""Active Hours Wrapper — skips inactive timesteps (night/unoccupied).

The wrapper internally fast-forwards through timesteps where the facade
has no influence (no solar radiation and/or no occupants), holding the
last action constant.  SB3 only sees transitions from active hours.

All timesteps are still simulated and logged by FradsEnv — the wrapper
only controls which transitions are exposed to the training loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class ActiveHoursWrapper(gym.Wrapper):
    """Skip inactive timesteps, exposing only active transitions to the agent.

    Parameters:
        env: Base FradsEnv (or wrapped FradsEnv).
        check_solar: If True, require GHI > 0 for a timestep to be active.
        check_occupancy: If True, require occupant_count > 0 to be active.
        mode: ``"or"`` — active if *either* condition is met (default).
              ``"and"`` — active only if *both* conditions are met.
        solar_keys: Info-dict keys (without ``raw_next_`` prefix) summed
            for the GHI check.  Default: DNI + DHI.
        occupancy_key: Info-dict key (without ``raw_next_`` prefix) for
            the occupancy check.
    """

    def __init__(
        self,
        env: gym.Env,
        check_solar: bool = True,
        check_occupancy: bool = False,
        mode: str = "or",
        solar_keys: Optional[List[str]] = None,
        occupancy_key: str = "occupant_count_1",
    ):
        super().__init__(env)
        self.check_solar = check_solar
        self.check_occupancy = check_occupancy
        self.mode = mode
        self.solar_keys = solar_keys or [
            "direct_normal_irradiance",
            "diffuse_horizontal_irradiance",
        ]
        self.occupancy_key = occupancy_key
        self._last_action: Optional[np.ndarray] = None
        self._default_action = np.zeros(env.action_space.shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_action = None

        # Simulation may start at midnight (inactive).
        # Fast-forward to first active timestep.
        while not self._is_active(info):
            info["agent_active"] = False
            obs, _reward, terminated, truncated, info = self.env.step(
                self._default_action
            )
            if terminated or truncated:
                break

        info["agent_active"] = True
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        # 1. Execute the agent's chosen action (active step).
        info_pre = getattr(self.env, "info", {})
        info_pre["agent_active"] = True
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # 2. Fast-forward through inactive timesteps.
        skipped = False
        while not self._is_active(info):
            skipped = True
            info["agent_active"] = False
            obs, _reward, terminated, truncated, info = self.env.step(
                self._last_action
            )
            if terminated or truncated:
                break

        info["agent_active"] = True
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_scalar(value) -> float:
        """Convert a scalar, 0-d array, or 1-element array to float."""
        arr = np.asarray(value)
        return float(arr.item())

    def _is_active(self, info: dict) -> bool:
        """Check whether the *next* timestep is active."""
        conditions = []

        if self.check_solar:
            ghi = sum(
                self._to_scalar(info.get(f"raw_next_{k}", 0))
                for k in self.solar_keys
            )
            conditions.append(ghi > 0)

        if self.check_occupancy:
            occ = self._to_scalar(info.get(f"raw_next_{self.occupancy_key}", 0))
            conditions.append(occ > 0)

        if not conditions:
            return True  # no checks enabled → always active

        if self.mode == "or":
            return any(conditions)
        else:  # "and"
            return all(conditions)
