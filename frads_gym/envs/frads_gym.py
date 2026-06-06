import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv
import pickle
import pandas as pd
from .frads_wrapper import FradsSimulation
import sys
import json
import math
import re

# Minimum observation history samples before switching to dynamic statistics.
# Convention: 96 steps = 1 day at 15-min intervals.
# Keep in sync with rewards.py and _reward_settings.py.
MIN_SAMPLES_DEFAULT = 96


class FradsEnv(gym.Env):
    """
    Gymnasium environment for FRADS building control simulation.
    
    This environment allows control of:
    - Facade shading (electrochromic glazing state)
    - Cooling setpoint temperature
    - Lighting power
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None, output_dir=None, config_file=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, run_periods=None, number_of_timesteps_per_hour=1, reward_function=None, logging=False, enable_radiance=True, truncate_on="none", staggered_start=False, env_id=0, n_envs=1, normalizer_state_dir=None):
        """
        Initialize the FRADS gymnasium environment

        
        Args:
            render_mode: Rendering mode (not fully implemented)
            config_file: Path to configuration file
        """
        self.render_mode = render_mode
        self.config_file = config_file
        self.output_dir = output_dir
        self.logging = logging
        self.info = {}
        self.reward_function = reward_function
        self.truncated_flag = False
        self.truncate_on = truncate_on
        # Stash for get_city_id's fallback path when self.simulation.weather_files_path is unset.
        self._weather_files_path = weather_files_path

        # Get config file directory for resolving relative paths and load config
        loaded_config = {}
        sim_config_arg = config_file

        if config_file:
            if isinstance(config_file, str):
                self.config_dir = os.path.dirname(os.path.abspath(config_file))
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Inject origin path
                loaded_config['config_dir'] = self.config_dir
                loaded_config['config_file_path'] = os.path.abspath(config_file)
                
                # Use the loaded dict for simulation to avoid reloading
                sim_config_arg = loaded_config
                
            elif isinstance(config_file, dict):
                loaded_config = config_file
                self.config_dir = loaded_config.get('config_dir', os.path.dirname(os.path.abspath(__file__)))
                sim_config_arg = loaded_config
        else:
            self.config_dir = os.path.dirname(os.path.abspath(__file__))

        if loaded_config:
            # Filter out inactive entries
            self.epsetup_config = {k: v for k, v in loaded_config.items() if isinstance(v, dict) and v.get("active", True)}
            print(f"Loaded {len(loaded_config)} config entries, kept {len(self.epsetup_config)} active entries")

        # Initialize the EnergyPlus simulation
        self.simulation = FradsSimulation(output_dir=output_dir, config_file=sim_config_arg, run_annual=run_annual, cleanup=cleanup, run_period=run_period, treat_weather_as_actual=treat_weather_as_actual, weather_files_path=weather_files_path, run_periods=run_periods, number_of_timesteps_per_hour=number_of_timesteps_per_hour, enable_radiance=enable_radiance, staggered_start=staggered_start, env_id=env_id, n_envs=n_envs)
        
        self.create_spaces_from_config()
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

        # Per-city observation-normalizer transfer. If a directory is
        # passed we look for normalizer_state_<city_id>.pkl inside it.
        # Missing file → cold-start (expected for zero-shot test cities
        # like Houston).
        #
        # Hard break: the legacy single-file `normalizer_state_path`
        # parameter is no longer supported. See spec 2026-04-21.
        if normalizer_state_dir:
            if not os.path.isdir(normalizer_state_dir):
                raise FileNotFoundError(
                    f"[FradsEnv env_id={env_id}] normalizer_state_dir was "
                    f"passed ({normalizer_state_dir}) but the directory does "
                    f"not exist."
                )
            city_id = self.extract_city_id(
                config_file=config_file,
                weather_files_path=weather_files_path,
            )
            self._city_id = city_id
            loaded = self._try_load_normalizer_from_dir(normalizer_state_dir, city_id)
            if loaded:
                print(f"[FradsEnv env_id={env_id}] loaded normalizer state "
                      f"for city='{city_id}' from {normalizer_state_dir}")
            else:
                print(f"[FradsEnv env_id={env_id}] no pickle for city='{city_id}' "
                      f"in {normalizer_state_dir} — cold-start (zero-shot)")

    # ------------------------------------------------------------------
    # Observation-normalizer state persistence (training → eval handoff)
    # ------------------------------------------------------------------

    _NORMALIZER_STATE_VERSION = 1

    # ------------------------------------------------------------------
    # City-identifier extraction (used to match training trackers to
    # eval envs for per-city normalizer transfer).
    # ------------------------------------------------------------------

    @staticmethod
    def extract_city_id(config_file, weather_files_path):
        """Return a canonical lowercase city identifier for this env.

        Matching order:
          1. weather_files_path[0].parent.name.lower() — the TMY directory
             (e.g., "Chicago" → "chicago").
          2. config_file["model_setup"]["model_path"] — regex looks for
             "_USA_<state>_<CITY>" segment before ".idf" (strips _MOD).
          3. ValueError if neither produces a city.

        Silent defaults are avoided on purpose: a wrong match would cause
        silent pickle mis-loading in downstream eval.
        """
        # Parent dir names that are NOT cities — e.g., bare filenames land in
        # the CWD whose basename could be anything, and the canonical weather
        # dir "TMY" is a sibling of per-city subdirs.
        _NON_CITY_PARENTS = {"tmy", "weather", ".", "..", ""}

        if weather_files_path:
            first = weather_files_path[0] if isinstance(weather_files_path, (list, tuple)) else weather_files_path
            if first:
                city = os.path.basename(os.path.dirname(os.path.abspath(str(first))))
                city_lower = city.lower()
                # Require a plausible city-dir name: starts with a letter, only
                # word chars afterwards, length >= 3, and not in the denylist.
                if (city_lower not in _NON_CITY_PARENTS
                        and re.match(r"^[a-z][a-z0-9_]{2,}$", city_lower)):
                    return city_lower

        if isinstance(config_file, dict):
            model_path = config_file.get("model_setup", {}).get("model_path", "")
        elif isinstance(config_file, str) and os.path.exists(config_file):
            with open(config_file, "r") as f:
                model_path = json.load(f).get("model_setup", {}).get("model_path", "")
        else:
            model_path = ""

        if model_path:
            # Strip extension and _MOD suffix, then take the trailing CITY token.
            stem = os.path.basename(model_path)
            stem = re.sub(r"\.idf$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"_MOD$", "", stem)
            # Require "_USA_<2letters>_<CITY>" at the end — avoids matching
            # arbitrary suffixes.
            m = re.search(r"_USA_[A-Z]{2}_([A-Z][A-Z_]*)$", stem)
            if m:
                return m.group(1).lower()

        raise ValueError(
            f"Cannot extract city_id: weather_files_path={weather_files_path!r}, "
            f"config_file model_path={model_path!r}. "
            f"Expected weather path like '.../Chicago/...epw' or IDF ending "
            f"in '_USA_XX_CITY.idf'."
        )

    def get_city_id(self) -> str:
        """Instance wrapper for use via SubprocVecEnv.env_method('get_city_id').

        Cached after first extraction via ``_city_id`` (set either in
        ``__init__`` when normalizer_state_dir is passed, or lazily here).
        """
        cached = getattr(self, "_city_id", None)
        if cached is not None:
            return cached
        city_id = self.extract_city_id(
            config_file=self.config_file,
            weather_files_path=getattr(self.simulation, "weather_files_path", None)
            or getattr(self, "_weather_files_path", None),
        )
        self._city_id = city_id
        return city_id

    def save_normalizer_state(self, path):
        """Pickle current observation-normalizer state to ``path``.

        Persists the stateful trackers that accumulate obs statistics:
          * ``_ema_trackers``            — EMA mean/var/count (used by ema_zscore)
          * ``stat_trackers``            — Welford for static z-score
          * ``observation_history``      — sample list for static min-max
          * ``observation_history_dynamic`` — sliding window for dyn-norm
          * ``_obs_welford_trackers``    — Welford used inside dyn-normaliser

        Static-config-only modes (raw, static_minmax with fixed bounds from
        JSON) don't rely on any of these; loading into them is a no-op.
        """
        state = {
            "version":       self._NORMALIZER_STATE_VERSION,
            "ema_trackers":  getattr(self, "_ema_trackers", {}),
            "stat_trackers": getattr(self, "stat_trackers", {}),
            "obs_history":   getattr(self, "observation_history", {}),
            "obs_history_dynamic":    getattr(self, "observation_history_dynamic", {}),
            "obs_welford_trackers":   getattr(self, "_obs_welford_trackers", {}),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
            f.flush()
            os.fsync(f.fileno())

    def load_normalizer_state(self, path):
        """Load observation-normalizer state from a pickle produced by
        :meth:`save_normalizer_state`. Overwrites any existing tracker state.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if the state-file schema version is incompatible.
            OSError / pickle.UnpicklingError: on corrupt files (propagated).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalizer state file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        version = state.get("version", 0)
        if version != self._NORMALIZER_STATE_VERSION:
            raise ValueError(
                f"Incompatible normalizer-state version {version} "
                f"(expected {self._NORMALIZER_STATE_VERSION}) in {path}"
            )
        self._ema_trackers              = state.get("ema_trackers", {})
        self.stat_trackers              = state.get("stat_trackers", {})
        self.observation_history        = state.get("obs_history", {})
        self.observation_history_dynamic = state.get("obs_history_dynamic", {})
        self._obs_welford_trackers      = state.get("obs_welford_trackers", {})

    def _try_load_normalizer_from_dir(self, normalizer_state_dir: str, city_id: str) -> bool:
        """Attempt to load a city-specific normalizer pickle from the given directory.

        Returns True if a matching `normalizer_state_<city_id>.pkl` was
        found and loaded, False if it was absent (caller should cold-start).
        Propagates any errors from :meth:`load_normalizer_state`.
        """
        pickle_path = os.path.join(
            normalizer_state_dir, f"normalizer_state_{city_id}.pkl"
        )
        if not os.path.exists(pickle_path):
            return False
        self.load_normalizer_state(pickle_path)
        return True


    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""

        if self.truncated_flag:
            # Episode was truncated mid-simulation (e.g. day/month boundary).
            # The EnergyPlus process is still running — just reset the flag
            # and continue stepping without restarting the simulation.
            # Normalization stats are preserved since the simulation is continuous.
            self.truncated_flag = False
        else:
            # Notify reward function that a full episode has ended
            # (skip on very first reset before any steps have been taken)
            if hasattr(self, 'reward_function') and self.reward_function is not None:
                if hasattr(self.reward_function, 'on_episode_end') and hasattr(self, 'raw_observation'):
                    self.reward_function.on_episode_end()

            # Full reset: restart EnergyPlus for a new episode.
            self.simulation.reset()

            # Observation normalization statistics persist across episode resets
            # (analogous to reward_history in RewardProcessingMixin).
            # This ensures stable normalization across multiple training episodes.
            # stat_trackers (Welford) and observation_history_dynamic are NOT reset.

        # Get initial observation (without taking an action)
        self.raw_next_obs = self.simulation.steps()

        # Process observation
        self.observation = self._process_observation(self.raw_next_obs)
        self.raw_observation = self.raw_next_obs

        # Reset per-episode step counter (raw_next_step_idx semantics: index of
        # the latest simulation step that produced raw_next_obs, starting at 0
        # for the initial observation returned by reset()).
        self._step_idx = 0

        # Return initial observation and info
        self.info['number_of_timesteps_per_hour'] = self.simulation.number_of_timesteps_per_hour
        # Expose active weather file index for per-climate reward baselines
        if hasattr(self.simulation, '_active_weather_idx'):
            self.info['weather_file_idx'] = self.simulation._active_weather_idx

        # Expose city_id + step_idx in info so reward functions (e.g. R1+ /
        # auto_rewards.energy_auto_r1p) can look up per-step oracle values
        # without relying on side-channels. List-wrapping matches the existing
        # raw_next_* convention.
        try:
            city_id = self.get_city_id()
        except Exception:
            city_id = "unknown"
        self.info['raw_next_city_id'] = [city_id]
        self.info['raw_next_step_idx'] = np.array([self._step_idx], dtype=np.int64)

        return self.observation, self.info
    
    def step(self, action):
        """
        Take a step in the environment by applying control actions.
        
        Args:
            action: Dict containing facade_state, cooling_setpoint and lighting_power
            
        Returns:
            observation: Next environment state
            reward: Reward for the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated (max steps)
            info: Additional information
        """
        # Convert gym action to simulation action format
        sim_action = self._process_action(action)
        
        # Add current observation to info with raw_current_ prefix BEFORE taking the step
        if hasattr(self, 'raw_observation') and self.raw_observation:
            for key, value in self.raw_observation.items():
                if not isinstance(value, (int, float, np.number)):
                    self.info[f'raw_current_{key}'] = value
                else:
                    self.info[f'raw_current_{key}'] = np.array([value], dtype=np.float32)
    
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)

        # A per-step obs-queue timeout (EnergyPlus callback stalled longer than the
        # queue timeout under multi-env resource contention) returns
        # {'error': ...} with NO observation keys. Passing it downstream KeyErrors
        # the reward and kills the whole SubprocVecEnv worker → the entire training
        # dies. Treat it as a hard episode end: truncate WITHOUT setting
        # truncated_flag, so the next reset() runs a full simulation.reset()
        # (restart EnergyPlus) and re-syncs the env↔EP step coupling.
        if isinstance(self.raw_next_obs, dict) and 'error' in self.raw_next_obs:
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype)
                                for key, space in self.observation_space.spaces.items()}
            self.info['sim_step_error'] = self.raw_next_obs.get('error')
            print(f"WARNING: per-step obs timeout/error -> truncating episode for "
                  f"EnergyPlus restart: {self.raw_next_obs.get('error')}")
            return next_observation, 0.0, False, True, self.info

        # Check if simulation is finished before processing further
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True # Episode is finished
            terminated = False # Not used in this context, if a terminal state is reached

            # Use zeros for next_observation
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype)
                              for key, space in self.observation_space.spaces.items()}

            # No reward at end of simulation
            reward = 0.0

            # Advance step counter so a reward function reading info after the
            # final step still sees a monotonically-increasing index.
            self._step_idx = getattr(self, '_step_idx', 0) + 1
            try:
                city_id = self.get_city_id()
            except Exception:
                city_id = "unknown"

            # Update info — include city/step keys so downstream consumers
            # (R1+ / oracle replay) can still identify the final step.
            info = {
                'simulation_finished': True,
                'raw_next_city_id': [city_id],
                'raw_next_step_idx': np.array([self._step_idx], dtype=np.int64),
            }
            # Also mirror onto self.info so callers inspecting env.info see
            # the same values.
            self.info['raw_next_city_id'] = info['raw_next_city_id']
            self.info['raw_next_step_idx'] = info['raw_next_step_idx']

            # print("Simulation finished, no further steps possible.")

            return next_observation, reward, terminated, truncated, info
        
        # Check for time period change
        if hasattr(self, 'truncate_on') and self._check_time_period_change(self.truncate_on):
            truncated = True  # End episode due to time period change
            terminated = False

            self.truncated_flag = True

            # Use current observation as the final observation
            next_observation = self._process_observation(self.raw_next_obs)

            # Advance step counter and expose city_id + step_idx so that the
            # final-step reward call below sees the same R1+ context as
            # regular steps.
            self._step_idx = getattr(self, '_step_idx', 0) + 1
            try:
                city_id = self.get_city_id()
            except Exception:
                city_id = "unknown"
            self.info['raw_next_city_id'] = [city_id]
            self.info['raw_next_step_idx'] = np.array([self._step_idx], dtype=np.int64)

            # Calculate reward for this last step
            reward = self.reward_function(self.observation, action, next_observation, self.info)

            # Update info
            self.info['time_period_change'] = True

            print("==="*50)
            print(f"Episode truncated due to time period change: {self.truncate_on}")
            print("==="*50)
            
            return next_observation, reward, terminated, truncated, self.info

        # If simulation is not finished, continue normal processing
        # Add each item from raw_next_obs to info with raw_next_ prefix
        for key, value in self.raw_next_obs.items():
            if not isinstance(value, (int, float, np.number)):
                self.info[f'raw_next_{key}'] = value
            else:
                self.info[f'raw_next_{key}'] = np.array([value], dtype=np.float32)

        # Advance per-episode step counter and expose city_id + step_idx for
        # reward functions that need to look up per-step oracle values (e.g.
        # auto_rewards.energy_auto_r1p / R1+). City id is cached after the
        # first call, so this is cheap.
        self._step_idx = getattr(self, '_step_idx', 0) + 1
        try:
            city_id = self.get_city_id()
        except Exception:
            city_id = "unknown"
        self.info['raw_next_city_id'] = [city_id]
        self.info['raw_next_step_idx'] = np.array([self._step_idx], dtype=np.int64)
        
        # Process observation from simulation
        next_observation = self._process_observation(self.raw_next_obs)
        
        # Calculate reward state, action, next_state
        # Skip reward computation during inactive steps (ActiveHoursWrapper fast-forward)
        # to avoid polluting reward normalisation statistics with night/unoccupied data.
        if self.info.get('agent_active', True):
            reward = self.reward_function(self.observation, action, next_observation, self.info)
        else:
            reward = 0.0

        # Log data
        if self.logging:
            self.log_data({
                'obs': self.raw_observation,
                'action': action,
                'next_obs': self.raw_next_obs,
                'next_obs_norm': next_observation,
                'reward': reward,
                'agent_active': self.info.get('agent_active', True),
            })

        # Update the current observation
        self.observation = next_observation
        self.raw_observation = self.raw_next_obs
        
        if self.render_mode == "raw_obs":
            print(self.raw_next_obs)
        elif self.render_mode == "raw_time":
            print(self.raw_next_obs['datetime'])

        truncated = False  # Only truncated if simulation signals completion
        terminated = False  # Not used in this context

        return next_observation, reward, terminated, truncated, self.info
    
    def close(self):
        """Clean up resources when environment is no longer needed."""
        if hasattr(self, 'simulation'):
            self.simulation.shutdown()
    
    def render(self):
        """
        Render the environment.
        """
        if not hasattr(self, 'raw_next_obs'):
            return "Environment not stepped yet"
        return str(self.raw_next_obs)

    def _process_observation(self, raw_obs):
        """
        Process raw simulation observations into gym observation space format.
        
        Args:
            raw_obs: Raw observation from simulation
            
        Returns:
            dict: Processed observation
        """
        # Safety check
        if raw_obs is None or 'error' in raw_obs or raw_obs.get('simulation_finished', False):
            # Return zeros for each observation space key
            return {key: np.zeros(space.shape, dtype=space.dtype) 
                    for key, space in self.observation_space.spaces.items()}
               
        # Initialize the processed observation dictionary
        processed_obs = {}
        
        # Process each key in the observation space
        for key in self.observation_space.spaces.keys():
            if key == "datetime" and "datetime" in raw_obs:
                # Convert datetime to normalized time of day (0-1)
                # Convert hour to radians (24 hours = 2π)
                # Process hour of day (24 hours = 2π)
                hour_rad = (raw_obs['datetime'].hour / 24.0) * 2 * np.pi
                hour_sin = np.sin(hour_rad)
                hour_cos = np.cos(hour_rad)
                
                # Process day of year (365/366 days = 2π)
                day_of_year = raw_obs['datetime'].timetuple().tm_yday
                days_in_year = 366 if raw_obs['datetime'].year % 4 == 0 else 365
                day_rad = (day_of_year / days_in_year) * 2 * np.pi
                day_sin = np.sin(day_rad)
                day_cos = np.cos(day_rad)

                # Process day of week (7 days = 2π)
                # Monday=0, Sunday=6
                day_of_week = raw_obs['datetime'].weekday()
                dow_rad = (day_of_week / 7.0) * 2 * np.pi
                dow_sin = np.sin(dow_rad)
                dow_cos = np.cos(dow_rad)

                # Store processed time of day
                processed_obs[key] = np.array([hour_sin, hour_cos, day_sin, day_cos, dow_sin, dow_cos], dtype=np.float32)

            elif "cfs_state" in key and key in raw_obs:
                # Map CFS state string to numeric value using facade_state_mapping
                cfs_value = self._map_facade_state_to_value(raw_obs[key])
                # Remap [0, 1] midpoints to the configured observation space bounds
                # (facade_state_mapping thresholds are left unchanged as they also drive action mapping)
                if hasattr(self, 'epsetup_config') and key in self.epsetup_config:
                    obs_cfg = self.epsetup_config[key].get("observation_space", {})
                    obs_low = obs_cfg.get("low", 0)
                    obs_high = obs_cfg.get("high", 1)
                    cfs_value = obs_low + cfs_value * (obs_high - obs_low)
                processed_obs[key] = np.array([cfs_value], dtype=np.float32)

            elif key in raw_obs:
                # Process keys that exist directly in raw_obs
                raw_value = raw_obs[key]
                
                # Check if the value is already an array/list
                if isinstance(raw_value, (list, tuple, np.ndarray)):
                    # Apply normalization to each element if needed
                    if hasattr(self, 'epsetup_config') and key in self.epsetup_config:
                        config = self.epsetup_config[key]
                        if "observation_normalize" in config and config["observation_normalize"]["active"]:
                            norm_low = np.array(config["observation_normalize"]["low"])
                            norm_high = np.array(config["observation_normalize"]["high"])
                            obs_low = np.array(config["observation_space"]["low"])
                            obs_high = np.array(config["observation_space"]["high"])
                            
                            # Normalize array values
                            raw_array = np.array(raw_value, dtype=np.float32)
                            normalized = (raw_array - norm_low) / (norm_high - norm_low)
                            processed_obs[key] = np.clip(normalized, obs_low, obs_high).astype(np.float32)
                        else:
                            processed_obs[key] = np.array(raw_value, dtype=np.float32)
                    else:
                        processed_obs[key] = np.array(raw_value, dtype=np.float32)
                else:
                    # For scalar values, normalize and wrap in array
                    normalized_value = self._normalize_observation(key, raw_value)
                    processed_obs[key] = np.array([normalized_value], dtype=np.float32)
            else:
                # Default for keys not found in raw_obs
                processed_obs[key] = np.zeros(self.observation_space.spaces[key].shape,
                                             dtype=self.observation_space.spaces[key].dtype)

        # Enforce the declared observation_space shape for every key. The scalar
        # standardisation path (ema_zscore) can return a (1,)-array which then
        # gets wrapped to (1,1); with SubprocVecEnv (n_envs>1) that stacks to
        # (n_envs,1,1) and breaks SB3's DictRolloutBuffer ("could not broadcast
        # (N,1,1) into (N,1)"). Reshaping to the space shape is a no-op when the
        # obs is already correct (size matches) and fixes the nested case.
        for _k, _space in self.observation_space.spaces.items():
            _arr = np.asarray(processed_obs[_k], dtype=_space.dtype)
            if _arr.shape != tuple(_space.shape) and _arr.size == int(np.prod(_space.shape)):
                _arr = _arr.reshape(_space.shape)
            processed_obs[_k] = _arr

        return processed_obs
    

    def _map_facade_state_to_value(self, state_name, facade_mapping=None):
        """
        Maps a facade state name (e.g., "ec01") to a normalized value (0-1) based on config mapping.
        
        Args:
            state_name (str): The facade state name (e.g., "ec01", "ec06", "ec18", "ec60")
            facade_mapping (dict, optional): The facade state mapping configuration. 
                                            If None, uses self.epsetup_config["facade_state_mapping"]
        
        Returns:
            float: Normalized value representing the facade state (0-1)
        """
        # Get the facade mapping configuration
        if facade_mapping is None and hasattr(self, 'epsetup_config'):
            facade_mapping = self.epsetup_config.get("facade_state_mapping", {})
        
        # If no mapping available or invalid, return default value
        if not facade_mapping or "states" not in facade_mapping:
            print("Warning: No valid facade state mapping found.")
            return 0.0
        
        # Find the state in the mapping
        for i, state in enumerate(facade_mapping["states"]):
            if state_name == state["name"]:
                # Use middle point of range for this state
                prev_threshold = 0.0 if i == 0 else facade_mapping["states"][i-1]["max_threshold"]
                current_threshold = state["max_threshold"]
                return prev_threshold + (current_threshold - prev_threshold) / 2
        
        # Return default if state not found
        print(f"Warning: Facade state '{state_name}' not found in mapping.")
        return 0.0


    def _map_value_to_facade_state(self, value, facade_mapping=None):
        """
        Maps a continuous value (0-1) to a discrete facade state based on config mapping.
        
        Args:
            value (float): A value between 0 and 1 representing the facade state
            facade_mapping (dict, optional): The facade state mapping configuration.
                                            If None, uses self.epsetup_config["facade_state_mapping"]
        
        Returns:
            str: The facade state code (e.g., "ec01", "ec06", "ec18", "ec60")
        """
        # Get the facade mapping configuration
        if facade_mapping is None and hasattr(self, 'epsetup_config'):
            facade_mapping = self.epsetup_config.get("facade_state_mapping", {})
        
        # If no mapping available or invalid, use default state
        if not facade_mapping or "states" not in facade_mapping or not facade_mapping["states"]:
            print("Warning: No valid facade state mapping found.")
            return ""
        
        # Ensure value is in valid range
        value = max(0.0, min(1.0, float(value)))
        
        # Find the appropriate state based on thresholds
        for state in facade_mapping["states"]:
            if value <= state["max_threshold"]:
                return state["name"]
        
        # Fallback to the last state if no match is found
        print(f"Warning: Value '{value}' exceeds all thresholds, using last state.")
        return facade_mapping["states"][-1]["name"]


    def _standardize_observation(self, key, value, config=None):
        """
        Standardize observation values using configurable methods with Z-scores.
        
        Supports two standardization approaches:
        1. Welford's algorithm for running statistics (real-time)
        2. Dynamic windowed statistics (history-based)
        
        Args:
            key: The observation key
            value: The raw observation value
            config: Optional config override for the key
            
        Returns:
            Standardized and normalized observation value
        """
        # If no config provided, find it
        if config is None:
            config, _ = self._find_key_in_config(key)
            if config is None:
                return value

        # Check if standardization is enabled
        if "observation_standardize" not in config or not config["observation_standardize"]["active"]:
            return value

        standardize_config = config["observation_standardize"]
        
        # Get observation space bounds for final normalization
        obs_low = config.get("observation_space", {}).get("low", 0)
        obs_high = config.get("observation_space", {}).get("high", 1)

        # Choose standardization method based on configuration
        dynamic_mode = standardize_config.get("dynamic", "welford")

        if dynamic_mode == "dynamic":
            return self._dynamic_windowed_standardization(key, value, standardize_config, obs_low, obs_high)
        elif dynamic_mode == "ema":
            return self._ema_standardization(key, value, standardize_config, obs_low, obs_high)
        else:
            return self._welford_standardization(key, value, standardize_config, obs_low, obs_high)


    def _ema_standardization(self, key, value, standardize_config, obs_low=0, obs_high=1):
        """Standardize using Exponential Moving Average (EMA) for mean and variance.

        Combines the O(1) efficiency of Welford with controlled forgetting:
        old samples decay exponentially, so the statistics adapt to distributional
        shifts (climate change, sim→hardware transfer) without saisonal window jumps.

        Config keys (inside ``observation_standardize``):
            ema_span (int): Effective window length in timesteps.  Default 35040
                (1 year at 15-min intervals).  Converted to α = 2 / (span + 1).
            min_samples (int): Welford warmup before switching to EMA.
            std_threshold (float): Floor for std to avoid division by zero.

        During warmup (count < min_samples) Welford statistics are used so the
        EMA doesn't start from uninformed priors.
        """
        if not hasattr(self, '_ema_trackers'):
            self._ema_trackers = {}

        if key not in self._ema_trackers:
            self._ema_trackers[key] = {
                'count': 0,
                'mean': 0.0,
                'var': 0.0,
                'std': 1.0,
                # Welford accumulators for warmup phase
                '_welford_mean': 0.0,
                '_welford_m2': 0.0,
            }

        t = self._ema_trackers[key]
        t['count'] += 1
        min_samples = standardize_config.get('min_samples', MIN_SAMPLES_DEFAULT)
        std_threshold = standardize_config.get('std_threshold', 1e-6)
        span = standardize_config.get('ema_span', 35040)  # default: 1 year
        alpha = 2.0 / (span + 1)

        # Always update Welford (needed for warmup, cheap O(1))
        delta_w = value - t['_welford_mean']
        t['_welford_mean'] += delta_w / t['count']
        delta_w2 = value - t['_welford_mean']
        t['_welford_m2'] += delta_w * delta_w2

        if t['count'] < min_samples:
            # Warmup: use Welford statistics
            if t['count'] > 1:
                variance = t['_welford_m2'] / t['count']
                t['std'] = max(np.sqrt(variance), std_threshold)
            t['mean'] = t['_welford_mean']
        elif t['count'] == min_samples:
            # Transition: seed EMA with Welford's final estimates
            t['mean'] = t['_welford_mean']
            t['var'] = t['_welford_m2'] / t['count']
            t['std'] = max(np.sqrt(t['var']), std_threshold)
        else:
            # EMA update
            delta = value - t['mean']
            t['mean'] += alpha * delta
            t['var'] = (1 - alpha) * (t['var'] + alpha * delta * delta)
            t['std'] = max(np.sqrt(t['var']), std_threshold)

        z_score = (value - t['mean']) / t['std']
        return self._apply_standardization_bounds(z_score, standardize_config, obs_low, obs_high)

    def _welford_standardization(self, key, value, standardize_config, obs_low=0, obs_high=1):
        """
        Standardize using Welford's online algorithm for running mean and variance.
        
        This method provides real-time statistics without storing history,
        making it memory-efficient for long-running simulations.
        
        Args:
            key: The observation key
            value: The raw observation value
            standardize_config: Standardization configuration
            obs_low: Lower bound of observation space
            obs_high: Upper bound of observation space
            
        Returns:
            Standardized value using running statistics
        """
        # Initialize statistics trackers if they don't exist
        if not hasattr(self, 'stat_trackers'):
            self.stat_trackers = {}
        
        if key not in self.stat_trackers:
            self.stat_trackers[key] = {
                'count': 0,
                'mean': 0.0,
                'var': 0.0,
                'std': 1.0
            }
    
        # Update running statistics using Welford's algorithm
        tracker = self.stat_trackers[key]
        tracker['count'] += 1
    
        # Welford's online algorithm
        delta = value - tracker['mean']
        tracker['mean'] += delta / tracker['count']
        delta2 = value - tracker['mean']
        tracker['var'] += delta * delta2
    
        # Calculate standard deviation with minimum threshold
        if tracker['count'] > 1:
            variance = tracker['var'] / tracker['count']
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            tracker['std'] = max(np.sqrt(variance), std_threshold)
    
        # Calculate Z-score
        z_score = (value - tracker['mean']) / tracker['std']
    
        # Apply bounds and normalize to observation space
        return self._apply_standardization_bounds(z_score, standardize_config, obs_low, obs_high)


    def _dynamic_windowed_standardization(self, key, value, standardize_config, obs_low=0, obs_high=1):
        """
        Standardize using dynamic windowed statistics from recent history.
    
        This method maintains a sliding window of recent values to compute
        statistics, allowing adaptation to changing conditions while considering
        minimum sample requirements.
        
        Args:
            key: The observation key
            value: The raw observation value
            standardize_config: Standardization configuration
            obs_low: Lower bound of observation space
            obs_high: Upper bound of observation space
            
        Returns:
            Standardized value using windowed statistics
        """
        # Initialize observation history if it doesn't exist
        if not hasattr(self, 'observation_history'):
            self.observation_history = {}
    
        if key not in self.observation_history:
            self.observation_history[key] = []
    
        # Store current value in history
        self.observation_history[key].append(value)
    
        # Configuration parameters
        min_samples = standardize_config.get('min_samples', MIN_SAMPLES_DEFAULT)
        sample_length = standardize_config.get('sample_length', 8640)
        std_threshold = standardize_config.get('std_threshold', 1e-4)

        # Maintain window size
        if len(self.observation_history[key]) > sample_length:
            self.observation_history[key] = self.observation_history[key][-sample_length:]

        # Update Welford tracker (always, O(1) cost)
        welford = self._update_obs_welford(key, value)

        # Check if we have enough samples for dynamic statistics
        history = self.observation_history[key]
        if len(history) >= min_samples:
            # Phase 2: Full dynamic — use window statistics
            mean_val = np.mean(history)
            std_val = np.std(history) if len(history) > 1 else 1.0
            std_val = max(std_val, std_threshold)
        elif welford['count'] >= 2:
            # Phase 1 (warmup): Blend static config → Welford estimate
            alpha = len(history) / min_samples
            static_mean = standardize_config.get('mean', 0.0)
            static_std = standardize_config.get('std', 1.0)
            mean_val = (1 - alpha) * static_mean + alpha * welford['mean']
            std_val = (1 - alpha) * static_std + alpha * welford['std']
            std_val = max(std_val, std_threshold)
        else:
            # Phase 0 (count < 2): Fall back to config defaults
            mean_val = standardize_config.get('mean', 0.0)
            std_val = standardize_config.get('std', 1.0)
    
        # Calculate Z-score
        z_score = (value - mean_val) / std_val
    
        # Debug output if enabled
        if standardize_config.get('debug', False):
            sample_info = f"({len(history)}/{min_samples} samples)"
            method = "dynamic" if len(history) >= min_samples else "fallback"
            print(f"Standardized {key} [{method}]: {z_score:.4f} -> obs_space[{obs_low},{obs_high}] {sample_info} "
                  f"(Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f})")
    
        # Apply bounds and normalize to observation space
        return self._apply_standardization_bounds(z_score, standardize_config, obs_low, obs_high)


    def _apply_clip(self, value, obs_low, obs_high, mode):
        """
        Apply clipping to a normalized value.

        Two modes are supported:
        - ``"hard"`` (default): hard clipping via ``np.clip``.
        - ``"soft"``           : tanh-based soft clipping that compresses
          values asymptotically toward the observation-space bounds without
          discarding the ordering of extreme events.  The value is first
          shifted so that [obs_low, obs_high] maps to [-1, 1], tanh is
          applied, then the result is shifted back.

        Args:
            value   : Normalized (but potentially out-of-range) scalar.
            obs_low : Lower bound of the target observation space.
            obs_high: Upper bound of the target observation space.
            mode    : ``"hard"`` | ``"soft"``.

        Returns:
            Clipped/compressed scalar value.
        """
        if mode == "soft":
            centre = (obs_high + obs_low) / 2.0
            half_range = (obs_high - obs_low) / 2.0
            if half_range == 0:
                return centre
            return centre + half_range * np.tanh((value - centre) / half_range)
        else:
            # Default: hard clipping
            return np.clip(value, obs_low, obs_high)

    def _apply_standardization_bounds(self, z_score, standardize_config, obs_low=0, obs_high=1):
        """
        Apply bounds to Z-score and optionally normalize to observation space.
        
        This helper function handles the final processing steps that are
        common to both standardization methods, mapping from standardization
        bounds to observation space bounds.
        
        Args:
            z_score: Calculated Z-score
            standardize_config: Standardization configuration
            obs_low: Lower bound of observation space
            obs_high: Upper bound of observation space
            
        Returns:
            Bounded and optionally normalized value for observation space
        """
        # Apply Z-score bounds
        std_low = standardize_config.get('low', -3)
        std_high = standardize_config.get('high', 3)
        
        # Handle extreme values with max_deviation if configured
        max_deviation = standardize_config.get('max_deviation', None)
        if max_deviation is not None:
            z_score = np.clip(z_score, -max_deviation, max_deviation)
        
        # Hard-clip to configured std bounds (always hard at this stage)
        clipped_z = np.clip(z_score, std_low, std_high)
        
        # Check if normalization to observation space is enabled
        normalize_to_obs_space = standardize_config.get('normalize_to_obs_space', True)
        
        if normalize_to_obs_space:
            # Normalize from standardization bounds to observation space bounds
            from_min, from_max = std_low, std_high
            to_min, to_max = obs_low, obs_high
            
            if from_min == from_max:
                return (to_min + to_max) / 2
            
            normalized = to_min + (clipped_z - from_min) * (to_max - to_min) / (from_max - from_min)
            clip_mode = standardize_config.get('clip_mode', 'hard')
            return self._apply_clip(normalized, to_min, to_max, clip_mode)
        else:
            # Return raw Z-score (clipped to std bounds)
            return clipped_z

    def _find_key_in_config(self, target_key, config_dict=None, path=None):
        """
        Recursively search for a target_key in nested configuration dictionaries.
        
        Args:
            target_key: The key to search for
            config_dict: The dictionary to search in (defaults to self.epsetup_config)
            path: Current path in the recursive search (for internal use)
            
        Returns:
            tuple: (found_config, path) where found_config is the configuration dict containing
                the key and path is the nested path to access it
        """
        if config_dict is None:
            config_dict = self.epsetup_config
        
        if path is None:
            path = []
        
        # First try direct match at current level
        if target_key in config_dict:
            return config_dict[target_key], path + [target_key]
        
        # Then search nested dictionaries
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Check if this nested dict has a 'name' that matches our target
                if "name" in value and value["name"] == target_key:
                    return value, path + [key]
                
                # Otherwise recursively search this dictionary
                found_config, found_path = self._find_key_in_config(target_key, value, path + [key])
                if found_config is not None:
                    return found_config, found_path
        
        # Not found
        return None, None

    def _normalize_observation(self, key, value):
        """
        Process observation values based on the configuration.
        Either normalizes to [0,1] range or standardizes (z-score) based on config.
        Now supports dynamic normalization similar to rewards.py
    
        Args:
            key: The observation key
            value: The raw observation value
            
        Returns:
            Processed observation value
        """
        # Special case for datetime which is handled separately
        if key == "datetime":
            return value
        
        # Find the configuration for this key, wherever it might be in the config structure
        config, path = self._find_key_in_config(key)
        
        if config is not None:
            # If we found a config entry with 'observation_space' that is active, use it
            if "observation_space" in config and config["observation_space"].get("active", True):
                obs_low = config["observation_space"]["low"]
                obs_high = config["observation_space"]["high"]
                
                # Check if standardization is active
                if "observation_standardize" in config and config["observation_standardize"]["active"]:
                    return self._standardize_observation(key, value, config=config)
                
                # Use normalization if standardization is not active
                elif "observation_normalize" in config and config["observation_normalize"]["active"]:
                    norm_config = config["observation_normalize"]
                    
                    # Check if dynamic normalization is enabled
                    dyn_mode = norm_config.get('dynamic', False)
                    if dyn_mode == "ema":
                        return self._ema_normalize_observation(key, value, norm_config, obs_low, obs_high)
                    elif dyn_mode:
                        return self._dynamic_normalize_observation(key, value, norm_config, obs_low, obs_high)
                    else:
                        # Use static normalization
                        return self._static_normalize_observation(key, value, norm_config, obs_low, obs_high)
    
        # If no normalization/standardization needed or config not found, return the original value
        return value

    def _static_normalize_observation(self, key, value, norm_config, obs_low, obs_high):
        """
        Static normalization using configured ranges
    
        Args:
            key: The observation key
            value: The raw observation value
            norm_config: Normalization configuration
            obs_low: Lower bound of observation space
            obs_high: Upper bound of observation space
            
        Returns:
            Normalized observation value
        """
        norm_low = norm_config["low"]
        norm_high = norm_config["high"]
        
        # Determine clip mode: explicit clip_mode takes precedence; fallback to
        # old boolean "clip" key for backwards compatibility.
        clip_mode = norm_config.get('clip_mode', None)
        if clip_mode is None and norm_config.get('clip', True):
            clip_mode = 'hard'

        # Handle array values
        if isinstance(norm_low, list):
            value_array = np.array(value)
            norm_low_array = np.array(norm_low)
            norm_high_array = np.array(norm_high)
            obs_low_array = np.array(obs_low)
            obs_high_array = np.array(obs_high)
            
            # Apply normalization element-wise
            normalized = obs_low_array + (value_array - norm_low_array) * (obs_high_array - obs_low_array) / (norm_high_array - norm_low_array)
            if clip_mode:
                normalized = np.array([self._apply_clip(v, lo, hi, clip_mode)
                                        for v, lo, hi in zip(normalized, obs_low_array, obs_high_array)],
                                       dtype=np.float32)
            return normalized
        else:
            # Handle scalar values
            if norm_high == norm_low:
                return (obs_low + obs_high) / 2
            normalized = obs_low + (value - norm_low) * (obs_high - obs_low) / (norm_high - norm_low)
            if clip_mode:
                normalized = self._apply_clip(normalized, obs_low, obs_high, clip_mode)
            return normalized

    def _update_obs_welford(self, key, value):
        """Update Welford running statistics for observation warmup blending."""
        if not hasattr(self, '_obs_welford_trackers'):
            self._obs_welford_trackers = {}
        if key not in self._obs_welford_trackers:
            self._obs_welford_trackers[key] = {'count': 0, 'mean': 0.0, 'M2': 0.0}
        t = self._obs_welford_trackers[key]
        t['count'] += 1
        delta = value - t['mean']
        t['mean'] += delta / t['count']
        delta2 = value - t['mean']
        t['M2'] += delta * delta2
        std = math.sqrt(t['M2'] / t['count']) if t['count'] > 1 else 1.0
        return {'mean': t['mean'], 'std': max(std, 1e-6), 'count': t['count']}

    def _dynamic_normalize_observation(self, key, value, norm_config, obs_low, obs_high):
        """
        Dynamic normalization using recent observation history
    
        Args:
            key: The observation key
            value: The raw observation value
            norm_config: Normalization configuration with dynamic settings
            obs_low: Lower bound of observation space
            obs_high: Upper bound of observation space
            
        Returns:
            Dynamically normalized observation value
        """
        # Initialize observation history if it doesn't exist
        if not hasattr(self, 'observation_history_dynamic'):
            self.observation_history_dynamic = {}
    
        if key not in self.observation_history_dynamic:
            self.observation_history_dynamic[key] = []
    
        # Store current value in history
        self.observation_history_dynamic[key].append(value)
    
        # Configuration for dynamic normalization
        min_samples_for_dynamic = norm_config.get('min_samples', MIN_SAMPLES_DEFAULT)
        sample_length = norm_config.get('sample_length', 8640)
        percentile_margin = norm_config.get('percentile_margin', 5.0)

        # Maintain window size
        if len(self.observation_history_dynamic[key]) > sample_length:
            self.observation_history_dynamic[key] = self.observation_history_dynamic[key][-sample_length:]

        # Update Welford tracker (always, O(1) cost)
        welford = self._update_obs_welford(key, value)

        # Use dynamic range if enough samples available
        history = self.observation_history_dynamic[key]
        if len(history) >= min_samples_for_dynamic:
            # Phase 2: Full dynamic — use percentiles for robust range estimation
            from_min = np.percentile(history, percentile_margin)
            from_max = np.percentile(history, 100 - percentile_margin)

            # Ensure min != max to avoid division by zero
            if from_min == from_max:
                # Add small margin around the constant value
                margin = abs(from_min) * 0.1 if from_min != 0 else 0.1
                from_min = from_min - margin
                from_max = from_max + margin

            # Debug output if requested
            if norm_config.get('debug', False):
                print(f"Dynamic normalization for {key}: "
                      f"Range [{from_min:.4f}, {from_max:.4f}] "
                      f"(from {len(history)} samples)")
        elif welford['count'] >= 2:
            # Phase 1 (warmup): Blend static config range → Welford estimate
            alpha = len(history) / min_samples_for_dynamic
            static_low = norm_config.get('low', 0)
            static_high = norm_config.get('high', 1)
            if isinstance(static_low, list):
                static_low = static_low[0] if len(static_low) > 0 else 0
            if isinstance(static_high, list):
                static_high = static_high[0] if len(static_high) > 0 else 1
            welford_low = welford['mean'] - 2 * welford['std']
            welford_high = welford['mean'] + 2 * welford['std']
            from_min = (1 - alpha) * static_low + alpha * welford_low
            from_max = (1 - alpha) * static_high + alpha * welford_high

            if norm_config.get('debug', False):
                print(f"Warmup normalization for {key}: "
                      f"Range [{from_min:.4f}, {from_max:.4f}] "
                      f"(alpha={alpha:.3f}, welford n={welford['count']})")
        else:
            # Phase 0 (count < 2): Fall back to static range from configuration
            from_min = norm_config.get('low', 0)
            from_max = norm_config.get('high', 1)

            # Handle array values for fallback
            if isinstance(from_min, list):
                from_min = from_min[0] if len(from_min) > 0 else 0
            if isinstance(from_max, list):
                from_max = from_max[0] if len(from_max) > 0 else 1
    
        # Perform normalization
        if from_max == from_min:
            normalized = (obs_low + obs_high) / 2
        else:
            normalized = obs_low + (value - from_min) * (obs_high - obs_low) / (from_max - from_min)
        
        # Apply clipping to observation space (clip_mode overrides legacy "clip" boolean).
        # "soft" uses tanh-based compression; "hard" uses np.clip.
        clip_mode = norm_config.get('clip_mode', None)
        if clip_mode is None and norm_config.get('clip', True):
            clip_mode = 'hard'
        if clip_mode:
            normalized = self._apply_clip(normalized, obs_low, obs_high, clip_mode)
        
        # Debug output if requested
        if norm_config.get('debug', False):
            method = "dynamic" if len(history) >= min_samples_for_dynamic else "fallback"
            print(f"Normalized {key} [{method}] [{clip_mode or 'no-clip'}]: {normalized:.4f} "
                  f"(Value: {value:.6f}, From: [{from_min:.6f}, {from_max:.6f}], "
                  f"To: [{obs_low:.6f}, {obs_high:.6f}])")
    
        return normalized

    def _ema_normalize_observation(self, key, value, norm_config, obs_low, obs_high):
        """Normalize using Exponential Moving Quantiles (EMQ) for P5/P95.

        O(1) per step — tracks P5 and P95 with asymmetric EMA updates
        (Frugal-2U style).  Adapts to distributional shifts without storing
        a history array.

        Config keys (inside ``observation_normalize``):
            ema_span (int): Effective window in timesteps.  Default 35040 (1 year).
                Converted to α = 2 / (span + 1).
            percentile_margin (float): Target quantile percentage.  Default 5.0
                → tracks P5 and P(100-5)=P95.
            min_samples (int): Welford warmup before switching to EMQ.
        """
        if not hasattr(self, '_ema_quantile_trackers'):
            self._ema_quantile_trackers = {}

        span = norm_config.get('ema_span', 17520)
        alpha = 2.0 / (span + 1)
        pct = norm_config.get('percentile_margin', 5.0) / 100.0  # e.g. 0.05
        min_samples = norm_config.get('min_samples', MIN_SAMPLES_DEFAULT)

        if key not in self._ema_quantile_trackers:
            # Seed from static config bounds
            static_low = norm_config.get('low', 0)
            static_high = norm_config.get('high', 1)
            if isinstance(static_low, list):
                static_low = static_low[0] if static_low else 0
            if isinstance(static_high, list):
                static_high = static_high[0] if static_high else 1
            self._ema_quantile_trackers[key] = {
                'p_low': float(static_low),
                'p_high': float(static_high),
                'count': 0,
            }

        t = self._ema_quantile_trackers[key]
        t['count'] += 1

        # Also update Welford for warmup blending
        welford = self._update_obs_welford(key, value)

        if t['count'] <= min_samples:
            # Warmup: blend static → Welford range
            blend = t['count'] / min_samples
            static_low = norm_config.get('low', 0)
            static_high = norm_config.get('high', 1)
            if isinstance(static_low, list):
                static_low = static_low[0] if static_low else 0
            if isinstance(static_high, list):
                static_high = static_high[0] if static_high else 1
            w_low = welford['mean'] - 2 * welford['std']
            w_high = welford['mean'] + 2 * welford['std']
            from_min = (1 - blend) * static_low + blend * w_low
            from_max = (1 - blend) * static_high + blend * w_high

            # Seed EMQ trackers at end of warmup
            if t['count'] == min_samples:
                t['p_low'] = from_min
                t['p_high'] = from_max
        else:
            # EMQ update — asymmetric step sizes push quantile toward target
            # P_low tracks the pct-quantile (e.g. P5)
            if value < t['p_low']:
                t['p_low'] += alpha * (value - t['p_low']) / pct
            else:
                t['p_low'] += alpha * (value - t['p_low']) / (1 - pct)

            # P_high tracks the (1-pct)-quantile (e.g. P95)
            if value < t['p_high']:
                t['p_high'] += alpha * (value - t['p_high']) / (1 - pct)
            else:
                t['p_high'] += alpha * (value - t['p_high']) / pct

            from_min = t['p_low']
            from_max = t['p_high']

        # Guard against degenerate range
        if from_max <= from_min:
            margin = abs(from_min) * 0.1 if from_min != 0 else 0.1
            from_max = from_min + margin

        # Normalize to obs space
        normalized = obs_low + (value - from_min) * (obs_high - obs_low) / (from_max - from_min)

        clip_mode = norm_config.get('clip_mode', None)
        if clip_mode is None and norm_config.get('clip', True):
            clip_mode = 'hard'
        if clip_mode:
            normalized = self._apply_clip(normalized, obs_low, obs_high, clip_mode)

        return normalized

    def _check_time_period_change(self, truncate_on="none"):
        """
        Check if the day or month has changed since the last observation and
        determines if the episode should be truncated based on the specified period.
        
        Args:
            truncate_on (str): The time period to check for changes. 
                            Options: "day", "month", "none" (default: "none")
        
        Returns:
            bool: True if the episode should be truncated based on the time period, False otherwise
        """
        # If not tracking time or truncate_on is none, don't truncate
        if truncate_on.lower() == "none" or "datetime" not in self.raw_next_obs:
            return False
        
        # Initialize previous_datetime if not exist
        if not hasattr(self, 'previous_datetime'):
            self.previous_datetime = self.raw_next_obs["datetime"]
            return False
        
        current_datetime = self.raw_next_obs["datetime"]
        previous_datetime = self.previous_datetime
        
        # Store current datetime for next comparison
        self.previous_datetime = current_datetime
        
        # Check for day change
        if truncate_on.lower() == "day":
            if (current_datetime.day != previous_datetime.day or 
                current_datetime.month != previous_datetime.month or 
                current_datetime.year != previous_datetime.year):
                return True
        
        # Check for month change
        elif truncate_on.lower() == "month":
            if (current_datetime.month != previous_datetime.month or 
                current_datetime.year != previous_datetime.year):
                return True
        
        return False

    def create_custom_spaces(self, config):
        """
        Dynamically create observation and action spaces based on the epsetup_config.
        
        Args:
            config: The loaded epsetup_config JSON
            
        Returns:
            tuple: (custom_observation_space, custom_action_space)
        """
        observation_dict = {
            # Always include datetime in observation space
            "datetime": spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        }
        
        # Track action space entries
        action_entries = {}
        
        # Process each entry in the config
        for var_id, var_info in config.items():
            # Add to observation space if it has observation_space defined and is active
            if "observation_space" in var_info and var_info["observation_space"].get("active", True):
                low = var_info["observation_space"]["low"]
                high = var_info["observation_space"]["high"]
                
                # Handle scalar vs array values
                if isinstance(low, list):
                    observation_dict[var_id] = spaces.Box(
                        low=np.array(low, dtype=np.float32),
                        high=np.array(high, dtype=np.float32),
                        dtype=np.float32
                    )
                else:
                    observation_dict[var_id] = spaces.Box(
                        low=np.array([low], dtype=np.float32),
                        high=np.array([high], dtype=np.float32),
                        shape=(1,),
                        dtype=np.float32
                    )
                
            # Track action spaces
            if "action_space" in var_info:
                action_entries[var_id] = {
                    "low": var_info["action_space"]["low"],
                    "high": var_info["action_space"]["high"]
                }
        
        # Create the observation space
        observation_space = spaces.Dict(observation_dict)
        
        # Create action space (Box with dimensions for each action)
        if action_entries:
            # Get all low and high values from action entries
            action_lows = []
            action_highs = []
            
            # Sort by key to ensure consistent order
            for var_id in sorted(action_entries.keys()):
                action_lows.append(action_entries[var_id]["low"])
                action_highs.append(action_entries[var_id]["high"])
            
            action_space = spaces.Box(
                low=np.array(action_lows, dtype=np.float32),
                high=np.array(action_highs, dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Default empty action space if none defined
            action_space = spaces.Box(
                low=np.array([0], dtype=np.float32),
                high=np.array([1], dtype=np.float32),
                dtype=np.float32
            )
        
        return observation_space, action_space

    def create_spaces_from_config(self):
        """
        Create observation and action spaces based on the epsetup_config.
        This should be called after epsetup_config is loaded.
        """
        if not hasattr(self, 'epsetup_config') or self.epsetup_config is None:
            # Load the configuration file if not already loaded
            config_path = os.path.join(self.input_dir, "epsetup_config.json")
            try:
                with open(config_path, 'r') as f:
                    self.epsetup_config = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Config file not found at {config_path}")
                self.epsetup_config = {}
        
        # Create spaces from config
        self.observation_space, self.action_space = self.create_custom_spaces(self.epsetup_config)

    def _process_action(self, action):
        """
        Convert gym action to simulation action format based on epsetup_config.
        
        Args:
            action: Action from gym environment
            
        Returns:
            dict: Action in simulation format with CFS states for each window
        """
        
        # Get all entries with actuate_cfs_state from config
        cfs_entries = {}
        idx = 0
        for var_id, var_info in self.epsetup_config.items():
            if "actuate_cfs_state" in var_info:
                cfs_entries[var_id] = self._map_value_to_facade_state(float(action[idx]))
                idx += 1
        
        return cfs_entries
    

    def log_data(self, data_dict):
        """
        Log environment data to a CSV file in a generalized way.
        
        Args:
            data_dict: Dictionary where keys are prefixes and values are data to log
        """
        log_path = os.path.join(self.output_dir, "environment_log.csv")
        file_exists = os.path.isfile(log_path)
        
        # Initialize file if not opened yet
        if not hasattr(self, 'log_file'):
            self.log_file = open(log_path, 'a', newline='')
            self.csv_writer = csv.writer(self.log_file)
        
        # Function to recursively extract values and generate column names
        def extract_data(prefix, data):
            columns = []
            values = []
            
            if isinstance(data, dict):
                # Handle dictionary data
                for k, v in data.items():
                    new_prefix = f"{prefix}_{k}"
                    if isinstance(v, (np.ndarray, list, tuple)) and not isinstance(v, str):
                        # For arrays in dictionaries, create a column for each element
                        flattened = np.array(v).flatten()
                        for i, item in enumerate(flattened):
                            columns.append(f"{new_prefix}_{i}")
                            values.append(format_value(item))
                    else:
                        # For non-array values
                        columns.append(new_prefix)
                        values.append(format_value(v))
            elif isinstance(data, (np.ndarray, list, tuple)) and not isinstance(data, str):
                # For array-like data, number each element
                flattened = np.array(data).flatten()
                for i, item in enumerate(flattened):
                    columns.append(f"{prefix}_{i}")
                    values.append(format_value(item))
            else:
                # For scalar data, just use the prefix
                columns.append(prefix)
                values.append(format_value(data))
            
            return columns, values
        
        # Helper function to format values appropriately
        def format_value(value):
            if hasattr(value, 'strftime'):  # Check if it's a datetime object
                return value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, (int, float, np.number)):
                return float(value)
            else:
                return value
        
        # Process all data to generate headers and values
        all_columns = []
        all_values = []
        
        for prefix, data in data_dict.items():
            columns, values = extract_data(prefix, data)
            all_columns.extend(columns)
            all_values.extend(values)
        
        # If file doesn't exist, write headers
        if not file_exists:
            self.csv_writer.writerow(all_columns)
        
        # Write the data row
        self.csv_writer.writerow(all_values)
        self.log_file.flush()  # Ensure data is written immediately