"""Solar forecast wrapper — adds short-horizon solar irradiance forecast.

Reads the EPW weather file and provides the next *forecast_horizon* timesteps
of Global Horizontal Irradiance (GHI) as:
  - An additional observation key ``solar_forecast`` (if add_to_obs=True)
  - An additional info key ``solar_forecast`` (always)

The forecast is "perfect" (deterministic from the EPW file), which is
epistemologically correct for simulation-based RL: the weather file IS
the ground truth.  For real deployment, replace with a weather API.

Synchronisation uses ``raw_next_datetime`` from the info dict (not a
fragile step counter).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def _parse_epw_ghi(epw_path: str) -> Dict[Tuple[int, int, int, int], float]:
    """Parse GHI (Global Horizontal Irradiance) from an EPW file.

    EPW format: 8 header lines, then hourly data rows.
    Column 13 (0-indexed) = Global Horizontal Radiation [Wh/m²].
    For sub-hourly timesteps, the hourly value is repeated.

    Returns:
        Dictionary mapping (month, day, hour, minute) → GHI [W/m²].
        For hourly EPW data, minute is always 0.
    """
    ghi_lookup: Dict[Tuple[int, int, int, int], float] = {}
    path = Path(epw_path)

    with open(path, "r") as f:
        # Skip 8 header lines
        for _ in range(8):
            next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 14:
                continue
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4]) if len(parts) > 4 else 0

            # EPW uses hour 1-24 convention; hour H covers the interval
            # (H-1):00–H:00.  Map uniformly to the 0-23 start-of-interval hour:
            # H → H-1 (so h1 → 0 … h24 → 23).  Mapping h24 → 0 would double-write
            # hour 0 (overwriting h1) and leave hour 23 empty.
            hour = hour - 1 if hour > 0 else 23

            ghi = float(parts[13])  # Global Horizontal Radiation [Wh/m²]
            # Wh/m² for 1h interval ≈ W/m² average over that hour
            ghi_lookup[(month, day, hour, minute)] = max(ghi, 0.0)

    return ghi_lookup


class SolarForecastWrapper(gym.Wrapper):
    """Add solar irradiance forecast to observations and info.

    Parameters:
        env: Base environment (must have Dict observation space).
        forecast_horizon: Number of future timesteps to forecast.
            Default 8 = 2 hours at 15-min timesteps.
        add_to_obs: If True, add ``solar_forecast`` to the observation
            space (Box, shape=(forecast_horizon,), normalised to [0, 1]).
        ghi_max: Normalisation ceiling for GHI [W/m²].  Default 1200.
        epw_path: Path to EPW file.  If None, extracted from the
            unwrapped environment at reset time.
    """

    def __init__(
        self,
        env: gym.Env,
        forecast_horizon: int = 8,
        add_to_obs: bool = True,
        ghi_max: float = 1200.0,
        epw_path: Optional[str] = None,
    ):
        super().__init__(env)
        self._forecast_horizon = forecast_horizon
        self._add_to_obs = add_to_obs
        self._ghi_max = ghi_max
        self._epw_path = epw_path
        self._ghi_lookup: Dict[Tuple[int, int, int, int], float] = {}
        self._timestep_minutes = 15  # EnergyPlus default

        # Extend observation space if requested
        if add_to_obs:
            assert isinstance(env.observation_space, gym.spaces.Dict), (
                "SolarForecastWrapper requires a Dict observation space"
            )
            new_spaces = dict(env.observation_space.spaces)
            new_spaces["solar_forecast"] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(forecast_horizon,),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Dict(new_spaces)

    def _sync_timestep_minutes(self, info: Dict) -> None:
        """Derive the timestep length from the env's actual timestep count.

        FradsEnv exposes ``number_of_timesteps_per_hour`` in info; using it keeps
        the forecast horizon in wall-clock terms correct (e.g. FTG 12/h = 5 min,
        not the 15-min default).  Falls back to 15 min if absent/invalid.
        """
        n = info.get("number_of_timesteps_per_hour")
        try:
            n = int(n)
            if n > 0:
                self._timestep_minutes = 60 / n
        except (TypeError, ValueError):
            pass

    def _load_epw(self) -> None:
        """Load EPW file from explicit path or from the unwrapped env."""
        epw = self._epw_path
        if epw is None:
            # Try to get from the simulation object
            base_env = self.unwrapped
            if hasattr(base_env, "simulation"):
                sim = base_env.simulation
                if hasattr(sim, "weather_files_path") and sim.weather_files_path:
                    # ``current_weather_idx`` is advanced modulo len AFTER the
                    # active file is chosen (frads_wrapper.py:590-591); on wrap
                    # (active = last file, current = 0) ``idx-1`` would give 0
                    # instead of len-1.  ``_active_weather_idx`` is the exact
                    # index of the file the current episode is running.
                    idx = getattr(sim, "_active_weather_idx", None)
                    if idx is None:
                        idx = max(0, getattr(sim, "current_weather_idx", 0) - 1)
                    epw = sim.weather_files_path[idx]
        if epw is not None:
            self._ghi_lookup = _parse_epw_ghi(epw)

    def _get_forecast(self, dt: datetime) -> np.ndarray:
        """Look up the next H timesteps of GHI from the parsed EPW data."""
        forecast = np.zeros(self._forecast_horizon, dtype=np.float32)
        for k in range(self._forecast_horizon):
            # Advance by (k+1) timesteps
            from datetime import timedelta
            future_dt = dt + timedelta(minutes=self._timestep_minutes * (k + 1))
            key = (future_dt.month, future_dt.day, future_dt.hour, 0)
            ghi = self._ghi_lookup.get(key, 0.0)
            forecast[k] = min(ghi / self._ghi_max, 1.0)
        return forecast

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._load_epw()
        self._sync_timestep_minutes(info)

        # Add zero forecast at reset (no datetime available yet)
        forecast = np.zeros(self._forecast_horizon, dtype=np.float32)
        info["solar_forecast"] = forecast
        if self._add_to_obs and isinstance(obs, dict):
            obs["solar_forecast"] = forecast

        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._sync_timestep_minutes(info)

        # Get forecast from datetime in info
        dt = info.get("raw_next_datetime", None)
        if dt is not None and self._ghi_lookup:
            forecast = self._get_forecast(dt)
        else:
            forecast = np.zeros(self._forecast_horizon, dtype=np.float32)

        info["solar_forecast"] = forecast
        if self._add_to_obs and isinstance(obs, dict):
            obs["solar_forecast"] = forecast

        return obs, reward, terminated, truncated, info
