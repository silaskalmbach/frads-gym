import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv
import pandas as pd
from frads_wrapper_test import EnergyPlusSimulation
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

    def __init__(self, render_mode=None, output_dir=None, input_dir=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, number_of_timesteps_per_hour=1, reward_function=None, logging=False, epsetup_config=None):
        """
        Initialize the FRADS gymnasium environment
        
        Args:
            render_mode: Rendering mode (not fully implemented)
            working_dir: Directory for simulation outputs
        """
        self.render_mode = render_mode
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.logging = logging
        self.info = {}
        self.reward_function = reward_function

        if epsetup_config is None:
            config_path = os.path.join(self.input_dir, "epsetup_config.json")
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Filter out inactive entries
            self.epsetup_config = {k: v for k, v in loaded_config.items() if v.get("active", True)}
            print(f"Loaded {len(loaded_config)} config entries, kept {len(self.epsetup_config)} active entries")
        else:
            # If config was provided externally, still filter inactive entries
            self.epsetup_config = {k: v for k, v in epsetup_config.items() if v.get("active", True)}

        # Initialize the EnergyPlus simulation
        self.simulation = EnergyPlusSimulation(output_dir=output_dir, input_dir=input_dir, run_annual=run_annual, cleanup=cleanup, run_period=run_period, treat_weather_as_actual=treat_weather_as_actual, weather_files_path=weather_files_path, number_of_timesteps_per_hour=number_of_timesteps_per_hour, epsetup_config=self.epsetup_config)
        
        self.create_spaces_from_config()
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")


    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        
        # Reset the simulation
        self.simulation.reset()
        
        # Reset statistics trackers for standardization
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
        # Convert gym action to simulation action format
        sim_action = self._process_action(action)
        
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)
        
        # Check if simulation is finished before processing further
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True
            terminated = False
            
            # Use zeros for next_observation
            next_observation = {key: np.zeros(space.shape, dtype=space.dtype) 
                              for key, space in self.observation_space.spaces.items()}
            
            # No reward at end of simulation
            reward = 0.0
            
            # Update info
            info = {'simulation_finished': True}
            
            return next_observation, reward, terminated, truncated, info
        
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
                normalized_value = self._normalize_observation(key, raw_obs[key])
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


    def _standardize_observation(self, key, value):
        """
        Standardize observation values using Z-scores and normalize to observation space.
        
        Args:
            key: The observation key
            value: The raw observation value
            
        Returns:
            Standardized and normalized observation value
        """
        if key not in self.epsetup_config:
            return value
            
        config = self.epsetup_config[key]
        
        # Check if standardization is enabled
        if "observation_standardize" not in config or not config["observation_standardize"]["active"]:
            return value
            
        # Initialize statistics trackers if they don't exist
        if not hasattr(self, 'stat_trackers'):
            self.stat_trackers = {}
            
        if key not in self.stat_trackers:
            # Initialize with a reasonable default for the first few steps
            self.stat_trackers[key] = {
                'count': 0,
                'mean': 0,
                'var': 0,  # To compute running variance
                'std': 1   # Default std to avoid division by zero
            }
        
        # Update running statistics using Welford's algorithm
        tracker = self.stat_trackers[key]
        tracker['count'] += 1
        delta = value - tracker['mean']
        tracker['mean'] += delta / tracker['count']
        delta2 = value - tracker['mean']
        tracker['var'] += delta * delta2
        
        # Calculate standard deviation with a minimum value to avoid division by zero
        if tracker['count'] > 1:
            tracker['std'] = max(np.sqrt(tracker['var'] / tracker['count']), 1e-8)
        
        # Calculate Z-score
        z_score = (value - tracker['mean']) / tracker['std']
        
        # Clip Z-score to the range specified in config
        std_low = config["observation_standardize"]["low"]
        std_high = config["observation_standardize"]["high"]
        clipped_z = np.clip(z_score, std_low, std_high)
        
        # Normalize to observation space
        obs_low = config["observation_space"]["low"]
        obs_high = config["observation_space"]["high"]
        
        # Map from std_low..std_high to obs_low..obs_high
        normalized_z = obs_low + (clipped_z - std_low) * (obs_high - obs_low) / (std_high - std_low)
        
        return normalized_z

    def _normalize_observation(self, key, value):
        """
        Process observation values based on the configuration.
        Either normalizes to [0,1] range or standardizes (z-score) based on config.
        
        Args:
            key: The observation key
            value: The raw observation value
            
        Returns:
            Processed observation value
        """
        # Special case for datetime which is handled separately
        if key == "datetime":
            return value
            
        # Check if this key exists in epsetup_config
        if key in self.epsetup_config:
            config = self.epsetup_config[key]
            
            # Get observation space bounds for clipping
            obs_low = config["observation_space"]["low"]
            obs_high = config["observation_space"]["high"]
            
            # Check if standardization is active (takes precedence over normalization)
            if "observation_standardize" in config and config["observation_standardize"]["active"]:
                # Use the new standardization function
                return self._standardize_observation(key, value)
            
            # Use normalization if standardization is not active
            elif "observation_normalize" in config:
                norm_low = config["observation_normalize"]["low"]
                norm_high = config["observation_normalize"]["high"]
                
                # Handle array values
                if isinstance(norm_low, list):
                    value_array = np.array(value)
                    norm_low_array = np.array(norm_low)
                    norm_high_array = np.array(norm_high)
                    obs_low_array = np.array(obs_low)
                    obs_high_array = np.array(obs_high)
                    
                    # Apply normalization and clip to observation space
                    normalized = (value_array - norm_low_array) / (norm_high_array - norm_low_array)
                    return np.clip(normalized, obs_low_array, obs_high_array)
                else:
                    # Handle scalar values
                    normalized = (value - norm_low) / (norm_high - norm_low)
                    return np.clip(normalized, obs_low, obs_high)
            
            # Special case for lighting_power
            elif "lighting_power" in config and key == config["lighting_power"]["name"]:
                if "observation_normalize" in config["lighting_power"]:
                    norm_low = config["lighting_power"]["observation_normalize"]["low"]
                    norm_high = config["lighting_power"]["observation_normalize"]["high"]
                    lighting_obs_low = config["lighting_power"]["observation_space"]["low"]
                    lighting_obs_high = config["lighting_power"]["observation_space"]["high"]
                    
                    normalized = (value - norm_low) / (norm_high - norm_low)
                    return np.clip(normalized, lighting_obs_low, lighting_obs_high)
        # Special case for lighting_power (search in all config entries)
        else:
            for var_id, var_info in self.epsetup_config.items():
                if "lighting_power" in var_info and key == var_info["lighting_power"]["name"]:
                    if "observation_normalize" in var_info["lighting_power"]:
                        norm_low = var_info["lighting_power"]["observation_normalize"]["low"]
                        norm_high = var_info["lighting_power"]["observation_normalize"]["high"]
                        lighting_obs_low = var_info["lighting_power"]["observation_space"]["low"]
                        lighting_obs_high = var_info["lighting_power"]["observation_space"]["high"]
                        
                        normalized = (value - norm_low) / (norm_high - norm_low)
                        return np.clip(normalized, lighting_obs_low, lighting_obs_high)
        
        # If no normalization/standardization needed or config not found, return the original value
        return value


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
            
            # Handle nested lighting_power observation space
            if "lighting_power" in var_info and "observation_space" in var_info["lighting_power"]:
                # Get the name for this lighting power entry
                lighting_power_name = var_info["lighting_power"]["name"]
                low = var_info["lighting_power"]["observation_space"]["low"]
                high = var_info["lighting_power"]["observation_space"]["high"]
                
                # Add to observation dictionary with its specified name
                if isinstance(low, list):
                    observation_dict[lighting_power_name] = spaces.Box(
                        low=np.array(low, dtype=np.float32),
                        high=np.array(high, dtype=np.float32),
                        dtype=np.float32
                    )
                else:
                    observation_dict[lighting_power_name] = spaces.Box(
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