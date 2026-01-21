import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv
import pandas as pd
from frads_wrapper import EnergyPlusSimulation
import sys
import json


class FradsEnv(gym.Env):
    """
    Gymnasium environment for FRADS building control simulation.
    
    This environment allows control of:
    - Facade shading (electrochromic glazing state)
    - Cooling setpoint temperature
    - Lighting power
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None, output_dir=None, config_file=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, number_of_timesteps_per_hour=1, reward_function=None, logging=False, enable_radiance=True, truncate_on="none"):
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
        
        # Get config file directory for resolving relative paths
        if config_file:
            self.config_dir = os.path.dirname(os.path.abspath(config_file))
        else:
            self.config_dir = os.path.dirname(os.path.abspath(__file__))

        with open(self.config_file, 'r') as f:
            loaded_config = json.load(f)
            
            # Filter out inactive entries
            self.epsetup_config = {k: v for k, v in loaded_config.items() if v.get("active", True)}
            print(f"Loaded {len(loaded_config)} config entries, kept {len(self.epsetup_config)} active entries")

        # Initialize the EnergyPlus simulation
        self.simulation = EnergyPlusSimulation(output_dir=output_dir, config_file=self.config_file, run_annual=run_annual, cleanup=cleanup, run_period=run_period, treat_weather_as_actual=treat_weather_as_actual, weather_files_path=weather_files_path, number_of_timesteps_per_hour=number_of_timesteps_per_hour, enable_radiance=enable_radiance)
        
        self.create_spaces_from_config()
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")


    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        
        if self.truncated_flag:
            # Reset the truncated flag
            self.truncated_flag = False
        else:
            # Reset the simulation
            self.simulation.reset()
        
        # Reset statistics trackers for both methods
        if hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
    
        if hasattr(self, 'observation_history'):
            self.observation_history = {}
    
        # Legacy stat_trackers (remove if not needed)
        if hasattr(self, 'stat_trackers'):
            self.stat_trackers = {}
        
        # Get initial observation (without taking an action)
        self.raw_next_obs = self.simulation.steps()
        
        # Process observation
        self.observation = self._process_observation(self.raw_next_obs)
        self.raw_observation = self.raw_next_obs
        
        # Return initial observation and info
        self.info['number_of_timesteps_per_hour'] = self.simulation.number_of_timesteps_per_hour

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
        # Add current state raw data to info with raw_current_ prefix
        if hasattr(self, 'raw_observation'):
            for key, value in self.raw_observation.items():
                if not isinstance(value, (int, float, np.number)):
                    self.info[f'raw_current_{key}'] = value
                else:
                    self.info[f'raw_current_{key}'] = np.array([value], dtype=np.float32)
        
        # Convert gym action to simulation action format
        sim_action = self._process_action(action)
        
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)
        
        # Check if simulation is finished before processing further
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True # Episode is finished
            terminated = False # Not used in this context, if a terminal state is reached
            
            # Use zeros for next_observation
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype) 
                              for key, space in self.observation_space.spaces.items()}
            
            # No reward at end of simulation
            reward = 0.0
            
            # Update info
            info = {'simulation_finished': True}

            # print("Simulation finished, no further steps possible.")
            
            return next_observation, reward, terminated, truncated, info
        
        if hasattr(self, 'truncate_on') and self._check_time_period_change(self.truncate_on):
            truncated = True  # End episode due to time period change
            terminated = False

            self.truncated_flag = True
            
            # Use current observation as the final observation
            next_observation = self._process_observation(self.raw_next_obs)
            
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
    
        # Process observation from simulation
        next_observation = self._process_observation(self.raw_next_obs)
        
        # Calculate reward state, action, next_state
        reward = self.reward_function(self.observation, action, next_observation, self.info)
        print(f"Reward: {reward}")

        # Log data
        if self.logging:
            self.log_data({
                'obs': self.raw_observation,
                'action': action,
                'next_obs': self.raw_next_obs,
                # 'info': self.info,
                'next_obs_norm': next_observation,
                'reward': reward
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

                # Store processed time of day              
                processed_obs[key] = np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)

            elif "cfs_state" in key and key in raw_obs:
                # Map CFS state string to numeric value using facade_state_mapping
                cfs_value = self._map_facade_state_to_value(raw_obs[key])
                # Store as a single-element array
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
        
        return processed_obs
    
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
            "datetime": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        }
        
        # Track action space entries
        action_entries = {}
        
        # Process each entry in the config
        for var_id, var_info in config.items():
            # Add to observation space if it has observation_space defined
            if "observation_space" in var_info:
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
        Standardize observation values using either dynamic windowing or Welford's algorithm.
        
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
    
        # Get standardization configuration
        standardize_config = config["observation_standardize"]
        method = standardize_config.get('method', 'dynamic')  # 'dynamic' or 'welford'
        min_samples_for_dynamic = standardize_config.get('min_samples', 50)
        std_threshold = standardize_config.get('std_threshold', 1e-6)
    
        # Default values from configuration
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
    
        if method == 'welford':
            # Use Welford's running algorithm (existing implementation)
            mean_val, std_val = self._welford_standardization(key, value, standardize_config)
        elif method == 'dynamic':
            # Use dynamic windowing approach
            mean_val, std_val = self._dynamic_windowing_standardization(key, value, standardize_config)
        else:
            print(f"Warning: Unknown standardization method '{method}', using fixed values")
    
        # Calculate z-score with robust handling of small standard deviations
        if std_val <= std_threshold:
            # Direct deviation without division for very small std
            z_score = value - mean_val
            # Limit extreme values
            max_deviation = standardize_config.get('max_deviation', 10.0)
            z_score = np.clip(z_score, -max_deviation, max_deviation)
        else:
            z_score = (value - mean_val) / std_val
    
        # Debug output (optional)
        if standardize_config.get('debug', False):
            method_info = f"method={method}"
            if std_val <= std_threshold:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.2e}, {method_info} - using direct deviation)")
            else:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}, {method_info})")
    
        # Clip z-score to configured range
        std_low = standardize_config["low"]
        std_high = standardize_config["high"]
        clipped_z = np.clip(z_score, std_low, std_high)
    
        # Normalize to observation space
        obs_low = config["observation_space"]["low"]
        obs_high = config["observation_space"]["high"]
    
        # Map from std_low..std_high to obs_low..obs_high
        normalized_z = obs_low + (clipped_z - std_low) * (obs_high - obs_low) / (std_high - std_low)
    
        return normalized_z

    def _welford_standardization(self, key, value, standardize_config):
        """
        Welford's running algorithm for online mean and variance calculation.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize statistics trackers if they don't exist
        if not hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
            
        if key not in self.welford_trackers:
            self.welford_trackers[key] = {
                'count': 0,
                'mean': standardize_config.get('mean', 0.0),
                'M2': 0.0,  # Sum of squares of differences from mean
                'std': standardize_config.get('std', 1.0)
            }
        
        # Update running statistics using Welford's algorithm
        tracker = self.welford_trackers[key]
        tracker['count'] += 1
        
        # Welford's algorithm
        delta = value - tracker['mean']
        tracker['mean'] += delta / tracker['count']
        delta2 = value - tracker['mean']
        tracker['M2'] += delta * delta2
        
        # Calculate standard deviation with minimum threshold
        if tracker['count'] > 1:
            variance = tracker['M2'] / tracker['count']
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            tracker['std'] = max(np.sqrt(variance), std_threshold)
        
        return tracker['mean'], tracker['std']

    def _dynamic_windowing_standardization(self, key, value, standardize_config):
        """
        Dynamic windowing approach for standardization statistics.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize observation history if it doesn't exist
        if not hasattr(self, 'observation_history'):
            self.observation_history = {}
        
        if key not in self.observation_history:
            self.observation_history[key] = []
        
        # Store current value in history
        self.observation_history[key].append(value)
        
        # Keep only recent history (configurable window)
        max_history = standardize_config.get('sample_length', 1000)
        if len(self.observation_history[key]) > max_history:
            self.observation_history[key] = self.observation_history[key][-max_history:]
        
        # Get default values
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
        min_samples = standardize_config.get('min_samples', 50)
        
        # Dynamic standardization if enough samples available
        if len(self.observation_history[key]) >= min_samples:
            recent_values = self.observation_history[key]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values) if len(recent_values) > 1 else 1.0
            
            # Prevent extremely small standard deviations
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            std_val = max(std_val, std_threshold)
        
        return mean_val, std_val

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        
        if self.truncated_flag:
            # Reset the truncated flag
            self.truncated_flag = False
        else:
            # Reset the simulation
            self.simulation.reset()
        
        # Reset statistics trackers for both methods
        if hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
    
        if hasattr(self, 'observation_history'):
            self.observation_history = {}
    
        # Legacy stat_trackers (remove if not needed)
        if hasattr(self, 'stat_trackers'):
            self.stat_trackers = {}
    
        # Get initial observation (without taking an action)
        self.raw_next_obs = self.simulation.steps()
        
        # Process observation
        self.observation = self._process_observation(self.raw_next_obs)
        self.raw_observation = self.raw_next_obs
        
        # Return initial observation and info
        self.info['number_of_timesteps_per_hour'] = self.simulation.number_of_timesteps_per_hour

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
        # Add current state raw data to info with raw_current_ prefix
        if hasattr(self, 'raw_observation'):
            for key, value in self.raw_observation.items():
                if not isinstance(value, (int, float, np.number)):
                    self.info[f'raw_current_{key}'] = value
                else:
                    self.info[f'raw_current_{key}'] = np.array([value], dtype=np.float32)
        
        # Convert gym action to simulation action format
        sim_action = self._process_action(action)
        
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)
        
        # Check if simulation is finished before processing further
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True # Episode is finished
            terminated = False # Not used in this context, if a terminal state is reached
            
            # Use zeros for next_observation
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype) 
                              for key, space in self.observation_space.spaces.items()}
            
            # No reward at end of simulation
            reward = 0.0
            
            # Update info
            info = {'simulation_finished': True}

            # print("Simulation finished, no further steps possible.")
            
            return next_observation, reward, terminated, truncated, info
        
        if hasattr(self, 'truncate_on') and self._check_time_period_change(self.truncate_on):
            truncated = True  # End episode due to time period change
            terminated = False

            self.truncated_flag = True
            
            # Use current observation as the final observation
            next_observation = self._process_observation(self.raw_next_obs)
            
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
    
        # Process observation from simulation
        next_observation = self._process_observation(self.raw_next_obs)
        
        # Calculate reward state, action, next_state
        reward = self.reward_function(self.observation, action, next_observation, self.info)
        print(f"Reward: {reward}")

        # Log data
        if self.logging:
            self.log_data({
                'obs': self.raw_observation,
                'action': action,
                'next_obs': self.raw_next_obs,
                # 'info': self.info,
                'next_obs_norm': next_observation,
                'reward': reward
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

                # Store processed time of day              
                processed_obs[key] = np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)

            elif "cfs_state" in key and key in raw_obs:
                # Map CFS state string to numeric value using facade_state_mapping
                cfs_value = self._map_facade_state_to_value(raw_obs[key])
                # Store as a single-element array
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
        Standardize observation values using either dynamic windowing or Welford's algorithm.
        
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
    
        # Get standardization configuration
        standardize_config = config["observation_standardize"]
        method = standardize_config.get('method', 'dynamic')  # 'dynamic' or 'welford'
        min_samples_for_dynamic = standardize_config.get('min_samples', 50)
        std_threshold = standardize_config.get('std_threshold', 1e-6)
    
        # Default values from configuration
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
    
        if method == 'welford':
            # Use Welford's running algorithm (existing implementation)
            mean_val, std_val = self._welford_standardization(key, value, standardize_config)
        elif method == 'dynamic':
            # Use dynamic windowing approach
            mean_val, std_val = self._dynamic_windowing_standardization(key, value, standardize_config)
        else:
            print(f"Warning: Unknown standardization method '{method}', using fixed values")
    
        # Calculate z-score with robust handling of small standard deviations
        if std_val <= std_threshold:
            # Direct deviation without division for very small std
            z_score = value - mean_val
            # Limit extreme values
            max_deviation = standardize_config.get('max_deviation', 10.0)
            z_score = np.clip(z_score, -max_deviation, max_deviation)
        else:
            z_score = (value - mean_val) / std_val
    
        # Debug output (optional)
        if standardize_config.get('debug', False):
            method_info = f"method={method}"
            if std_val <= std_threshold:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.2e}, {method_info} - using direct deviation)")
            else:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}, {method_info})")
    
        # Clip z-score to configured range
        std_low = standardize_config["low"]
        std_high = standardize_config["high"]
        clipped_z = np.clip(z_score, std_low, std_high)
    
        # Normalize to observation space
        obs_low = config["observation_space"]["low"]
        obs_high = config["observation_space"]["high"]
    
        # Map from std_low..std_high to obs_low..obs_high
        normalized_z = obs_low + (clipped_z - std_low) * (obs_high - obs_low) / (std_high - std_low)
    
        return normalized_z

    def _welford_standardization(self, key, value, standardize_config):
        """
        Welford's running algorithm for online mean and variance calculation.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize statistics trackers if they don't exist
        if not hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
            
        if key not in self.welford_trackers:
            self.welford_trackers[key] = {
                'count': 0,
                'mean': standardize_config.get('mean', 0.0),
                'M2': 0.0,  # Sum of squares of differences from mean
                'std': standardize_config.get('std', 1.0)
            }
        
        # Update running statistics using Welford's algorithm
        tracker = self.welford_trackers[key]
        tracker['count'] += 1
        
        # Welford's algorithm
        delta = value - tracker['mean']
        tracker['mean'] += delta / tracker['count']
        delta2 = value - tracker['mean']
        tracker['M2'] += delta * delta2
        
        # Calculate standard deviation with minimum threshold
        if tracker['count'] > 1:
            variance = tracker['M2'] / tracker['count']
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            tracker['std'] = max(np.sqrt(variance), std_threshold)
        
        return tracker['mean'], tracker['std']

    def _dynamic_windowing_standardization(self, key, value, standardize_config):
        """
        Dynamic windowing approach for standardization statistics.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize observation history if it doesn't exist
        if not hasattr(self, 'observation_history'):
            self.observation_history = {}
        
        if key not in self.observation_history:
            self.observation_history[key] = []
        
        # Store current value in history
        self.observation_history[key].append(value)
        
        # Keep only recent history (configurable window)
        max_history = standardize_config.get('sample_length', 1000)
        if len(self.observation_history[key]) > max_history:
            self.observation_history[key] = self.observation_history[key][-max_history:]
        
        # Get default values
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
        min_samples = standardize_config.get('min_samples', 50)
        
        # Dynamic standardization if enough samples available
        if len(self.observation_history[key]) >= min_samples:
            recent_values = self.observation_history[key]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values) if len(recent_values) > 1 else 1.0
            
            # Prevent extremely small standard deviations
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            std_val = max(std_val, std_threshold)
        
        return mean_val, std_val

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        
        if self.truncated_flag:
            # Reset the truncated flag
            self.truncated_flag = False
        else:
            # Reset the simulation
            self.simulation.reset()
        
        # Reset statistics trackers for both methods
        if hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
    
        if hasattr(self, 'observation_history'):
            self.observation_history = {}
    
        # Legacy stat_trackers (remove if not needed)
        if hasattr(self, 'stat_trackers'):
            self.stat_trackers = {}
    
        # Get initial observation (without taking an action)
        self.raw_next_obs = self.simulation.steps()
        
        # Process observation
        self.observation = self._process_observation(self.raw_next_obs)
        self.raw_observation = self.raw_next_obs
        
        # Return initial observation and info
        self.info['number_of_timesteps_per_hour'] = self.simulation.number_of_timesteps_per_hour

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
        # Add current state raw data to info with raw_current_ prefix
        if hasattr(self, 'raw_observation'):
            for key, value in self.raw_observation.items():
                if not isinstance(value, (int, float, np.number)):
                    self.info[f'raw_current_{key}'] = value
                else:
                    self.info[f'raw_current_{key}'] = np.array([value], dtype=np.float32)
        
        # Convert gym action to simulation action format
        sim_action = self._process_action(action)
        
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)
        
        # Check if simulation is finished before processing further
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True # Episode is finished
            terminated = False # Not used in this context, if a terminal state is reached
            
            # Use zeros for next_observation
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype) 
                              for key, space in self.observation_space.spaces.items()}
            
            # No reward at end of simulation
            reward = 0.0
            
            # Update info
            info = {'simulation_finished': True}

            # print("Simulation finished, no further steps possible.")
            
            return next_observation, reward, terminated, truncated, info
        
        if hasattr(self, 'truncate_on') and self._check_time_period_change(self.truncate_on):
            truncated = True  # End episode due to time period change
            terminated = False

            self.truncated_flag = True
            
            # Use current observation as the final observation
            next_observation = self._process_observation(self.raw_next_obs)
            
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
    
        # Process observation from simulation
        next_observation = self._process_observation(self.raw_next_obs)
        
        # Calculate reward state, action, next_state
        reward = self.reward_function(self.observation, action, next_observation, self.info)
        print(f"Reward: {reward}")

        # Log data
        if self.logging:
            self.log_data({
                'obs': self.raw_observation,
                'action': action,
                'next_obs': self.raw_next_obs,
                # 'info': self.info,
                'next_obs_norm': next_observation,
                'reward': reward
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

                # Store processed time of day              
                processed_obs[key] = np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)

            elif "cfs_state" in key and key in raw_obs:
                # Map CFS state string to numeric value using facade_state_mapping
                cfs_value = self._map_facade_state_to_value(raw_obs[key])
                # Store as a single-element array
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
        Standardize observation values using either dynamic windowing or Welford's algorithm.
        
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
    
        # Get standardization configuration
        standardize_config = config["observation_standardize"]
        method = standardize_config.get('method', 'dynamic')  # 'dynamic' or 'welford'
        min_samples_for_dynamic = standardize_config.get('min_samples', 50)
        std_threshold = standardize_config.get('std_threshold', 1e-6)
    
        # Default values from configuration
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
    
        if method == 'welford':
            # Use Welford's running algorithm (existing implementation)
            mean_val, std_val = self._welford_standardization(key, value, standardize_config)
        elif method == 'dynamic':
            # Use dynamic windowing approach
            mean_val, std_val = self._dynamic_windowing_standardization(key, value, standardize_config)
        else:
            print(f"Warning: Unknown standardization method '{method}', using fixed values")
    
        # Calculate z-score with robust handling of small standard deviations
        if std_val <= std_threshold:
            # Direct deviation without division for very small std
            z_score = value - mean_val
            # Limit extreme values
            max_deviation = standardize_config.get('max_deviation', 10.0)
            z_score = np.clip(z_score, -max_deviation, max_deviation)
        else:
            z_score = (value - mean_val) / std_val
    
        # Debug output (optional)
        if standardize_config.get('debug', False):
            method_info = f"method={method}"
            if std_val <= std_threshold:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.2e}, {method_info} - using direct deviation)")
            else:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}, {method_info})")
    
        # Clip z-score to configured range
        std_low = standardize_config["low"]
        std_high = standardize_config["high"]
        clipped_z = np.clip(z_score, std_low, std_high)
    
        # Normalize to observation space
        obs_low = config["observation_space"]["low"]
        obs_high = config["observation_space"]["high"]
    
        # Map from std_low..std_high to obs_low..obs_high
        normalized_z = obs_low + (clipped_z - std_low) * (obs_high - obs_low) / (std_high - std_low)
    
        return normalized_z

    def _welford_standardization(self, key, value, standardize_config):
        """
        Welford's running algorithm for online mean and variance calculation.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize statistics trackers if they don't exist
        if not hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
            
        if key not in self.welford_trackers:
            self.welford_trackers[key] = {
                'count': 0,
                'mean': standardize_config.get('mean', 0.0),
                'M2': 0.0,  # Sum of squares of differences from mean
                'std': standardize_config.get('std', 1.0)
            }
        
        # Update running statistics using Welford's algorithm
        tracker = self.welford_trackers[key]
        tracker['count'] += 1
        
        # Welford's algorithm
        delta = value - tracker['mean']
        tracker['mean'] += delta / tracker['count']
        delta2 = value - tracker['mean']
        tracker['M2'] += delta * delta2
        
        # Calculate standard deviation with minimum threshold
        if tracker['count'] > 1:
            variance = tracker['M2'] / tracker['count']
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            tracker['std'] = max(np.sqrt(variance), std_threshold)
        
        return tracker['mean'], tracker['std']

    def _dynamic_windowing_standardization(self, key, value, standardize_config):
        """
        Dynamic windowing approach for standardization statistics.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize observation history if it doesn't exist
        if not hasattr(self, 'observation_history'):
            self.observation_history = {}
        
        if key not in self.observation_history:
            self.observation_history[key] = []
        
        # Store current value in history
        self.observation_history[key].append(value)
        
        # Keep only recent history (configurable window)
        max_history = standardize_config.get('sample_length', 1000)
        if len(self.observation_history[key]) > max_history:
            self.observation_history[key] = self.observation_history[key][-max_history:]
        
        # Get default values
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
        min_samples = standardize_config.get('min_samples', 50)
        
        # Dynamic standardization if enough samples available
        if len(self.observation_history[key]) >= min_samples:
            recent_values = self.observation_history[key]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values) if len(recent_values) > 1 else 1.0
            
            # Prevent extremely small standard deviations
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            std_val = max(std_val, std_threshold)
        
        return mean_val, std_val

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        
        if self.truncated_flag:
            # Reset the truncated flag
            self.truncated_flag = False
        else:
            # Reset the simulation
            self.simulation.reset()
        
        # Reset statistics trackers for both methods
        if hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
    
        if hasattr(self, 'observation_history'):
            self.observation_history = {}
    
        # Legacy stat_trackers (remove if not needed)
        if hasattr(self, 'stat_trackers'):
            self.stat_trackers = {}
    
        # Get initial observation (without taking an action)
        self.raw_next_obs = self.simulation.steps()
        
        # Process observation
        self.observation = self._process_observation(self.raw_next_obs)
        self.raw_observation = self.raw_next_obs
        
        # Return initial observation and info
        self.info['number_of_timesteps_per_hour'] = self.simulation.number_of_timesteps_per_hour

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
        # Add current state raw data to info with raw_current_ prefix
        if hasattr(self, 'raw_observation'):
            for key, value in self.raw_observation.items():
                if not isinstance(value, (int, float, np.number)):
                    self.info[f'raw_current_{key}'] = value
                else:
                    self.info[f'raw_current_{key}'] = np.array([value], dtype=np.float32)
        
        # Convert gym action to simulation action format
        sim_action = self._process_action(action)
        
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)
        
        # Check if simulation is finished before processing further
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True # Episode is finished
            terminated = False # Not used in this context, if a terminal state is reached
            
            # Use zeros for next_observation
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype) 
                              for key, space in self.observation_space.spaces.items()}
            
            # No reward at end of simulation
            reward = 0.0
            
            # Update info
            info = {'simulation_finished': True}

            # print("Simulation finished, no further steps possible.")
            
            return next_observation, reward, terminated, truncated, info
        
        if hasattr(self, 'truncate_on') and self._check_time_period_change(self.truncate_on):
            truncated = True  # End episode due to time period change
            terminated = False

            self.truncated_flag = True
            
            # Use current observation as the final observation
            next_observation = self._process_observation(self.raw_next_obs)
            
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
    
        # Process observation from simulation
        next_observation = self._process_observation(self.raw_next_obs)
        
        # Calculate reward state, action, next_state
        reward = self.reward_function(self.observation, action, next_observation, self.info)
        print(f"Reward: {reward}")

        # Log data
        if self.logging:
            self.log_data({
                'obs': self.raw_observation,
                'action': action,
                'next_obs': self.raw_next_obs,
                # 'info': self.info,
                'next_obs_norm': next_observation,
                'reward': reward
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

                # Store processed time of day              
                processed_obs[key] = np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)

            elif "cfs_state" in key and key in raw_obs:
                # Map CFS state string to numeric value using facade_state_mapping
                cfs_value = self._map_facade_state_to_value(raw_obs[key])
                # Store as a single-element array
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
        Standardize observation values using either dynamic windowing or Welford's algorithm.
        
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
    
        # Get standardization configuration
        standardize_config = config["observation_standardize"]
        method = standardize_config.get('method', 'dynamic')  # 'dynamic' or 'welford'
        min_samples_for_dynamic = standardize_config.get('min_samples', 50)
        std_threshold = standardize_config.get('std_threshold', 1e-6)
    
        # Default values from configuration
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
    
        if method == 'welford':
            # Use Welford's running algorithm (existing implementation)
            mean_val, std_val = self._welford_standardization(key, value, standardize_config)
        elif method == 'dynamic':
            # Use dynamic windowing approach
            mean_val, std_val = self._dynamic_windowing_standardization(key, value, standardize_config)
        else:
            print(f"Warning: Unknown standardization method '{method}', using fixed values")
    
        # Calculate z-score with robust handling of small standard deviations
        if std_val <= std_threshold:
            # Direct deviation without division for very small std
            z_score = value - mean_val
            # Limit extreme values
            max_deviation = standardize_config.get('max_deviation', 10.0)
            z_score = np.clip(z_score, -max_deviation, max_deviation)
        else:
            z_score = (value - mean_val) / std_val
    
        # Debug output (optional)
        if standardize_config.get('debug', False):
            method_info = f"method={method}"
            if std_val <= std_threshold:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.2e}, {method_info} - using direct deviation)")
            else:
                print(f"Standardized obs {key}: {z_score:.4f} (Value: {value:.6f}, Mean: {mean_val:.6f}, Std: {std_val:.6f}, {method_info})")
    
        # Clip z-score to configured range
        std_low = standardize_config["low"]
        std_high = standardize_config["high"]
        clipped_z = np.clip(z_score, std_low, std_high)
    
        # Normalize to observation space
        obs_low = config["observation_space"]["low"]
        obs_high = config["observation_space"]["high"]
    
        # Map from std_low..std_high to obs_low..obs_high
        normalized_z = obs_low + (clipped_z - std_low) * (obs_high - obs_low) / (std_high - std_low)
    
    
        return normalized_z

    def _welford_standardization(self, key, value, standardize_config):
        """
        Welford's running algorithm for online mean and variance calculation.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize statistics trackers if they don't exist
        if not hasattr(self, 'welford_trackers'):
            self.welford_trackers = {}
            
        if key not in self.welford_trackers:
            self.welford_trackers[key] = {
                'count': 0,
                'mean': standardize_config.get('mean', 0.0),
                'M2': 0.0,  # Sum of squares of differences from mean
                'std': standardize_config.get('std', 1.0)
            }
        
        # Update running statistics using Welford's algorithm
        tracker = self.welford_trackers[key]
        tracker['count'] += 1
        
        # Welford's algorithm
        delta = value - tracker['mean']
        tracker['mean'] += delta / tracker['count']
        delta2 = value - tracker['mean']
        tracker['M2'] += delta * delta2
        
        # Calculate standard deviation with minimum threshold
        if tracker['count'] > 1:
            variance = tracker['M2'] / tracker['count']
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            tracker['std'] = max(np.sqrt(variance), std_threshold)
        
        return tracker['mean'], tracker['std']

    def _dynamic_windowing_standardization(self, key, value, standardize_config):
        """
        Dynamic windowing approach for standardization statistics.
        
        Args:
            key: The observation key
            value: Current observation value
            standardize_config: Configuration for standardization
            
        Returns:
            tuple: (mean, std)
        """
        # Initialize observation history if it doesn't exist
        if not hasattr(self, 'observation_history'):
            self.observation_history = {}
        
        if key not in self.observation_history:
            self.observation_history[key] = []
        
        # Store current value in history
        self.observation_history[key].append(value)
        
        # Keep only recent history (configurable window)
        max_history = standardize_config.get('sample_length', 1000)
        if len(self.observation_history[key]) > max_history:
            self.observation_history[key] = self.observation_history[key][-max_history:]
        
        # Get default values
        mean_val = standardize_config.get('mean', 0.0)
        std_val = standardize_config.get('std', 1.0)
        min_samples = standardize_config.get('min_samples', 50)
        
        # Dynamic standardization if enough samples available
        if len(self.observation_history[key]) >= min_samples:
            recent_values = self.observation_history[key]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values) if len(recent_values) > 1 else 1.0
            
            # Prevent extremely small standard deviations
            std_threshold = standardize_config.get('std_threshold', 1e-6)
            std_val = max(std_val, std_threshold)
        
        return mean_val, std_val
