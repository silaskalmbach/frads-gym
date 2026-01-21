import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import csv
import pandas as pd

import sys


class FradsEnv(gym.Env):
    """
    Gymnasium environment for FRADS building control simulation.
    
    This environment allows control of:
    - Facade shading (electrochromic glazing state)
    - Cooling setpoint temperature
    - Lighting power
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None, output_dir=None, input_dir=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, number_of_timesteps_per_hour=1, reward_function=None, logging=False):
        """
        Initialize the FRADS gymnasium environment
        
        Args:
            render_mode: Rendering mode (not fully implemented)
            working_dir: Directory for simulation outputs
        """
        self.render_mode = render_mode
        self.output_dir = output_dir
        self.logging = logging
        self.info = {}

        if reward_function is not None:
            print("Using custom reward function")
            self.reward_function = reward_function
        else:
            print("Using default reward function")
            self.reward_function = self._calculate_reward

        # Check if setup.py exists in input_dir to determine import method
        if input_dir and os.path.exists(os.path.join(input_dir, "custom_setup.py")):
            sys.path.append(input_dir)
            from custom_setup import EnergyPlusSimulation, custom_observation_space, custom_action_space, process_custom_action
            print("Using custom_setup.py for EnergyPlusSimulation import")
        else:
            # Use the standard import if no custom_setup.py is found
            sys.path.append(os.path.dirname(__file__))
            from frads_wrapper_test import EnergyPlusSimulation
            print("Using standard EnergyPlusSimulation import")

        if 'custom_action_space' in locals():
            self.process_action = process_custom_action
            print("Using custom action processing function")
        else:
            self.process_action = self._process_action
            print("Using default action processing function")

        # Initialize the EnergyPlus simulation
        self.simulation = EnergyPlusSimulation(output_dir=output_dir, input_dir=input_dir, run_annual=run_annual, cleanup=cleanup, run_period=run_period, treat_weather_as_actual=treat_weather_as_actual, weather_files_path=weather_files_path, number_of_timesteps_per_hour=number_of_timesteps_per_hour)
        
        # Check if custom spaces were defined in the imported module
        if 'custom_observation_space' in locals():
            self.observation_space = custom_observation_space
            print("Using custom observation space")
        else:
            # Define observation space
            self.observation_space = spaces.Dict({
                # External solar irradiance (W/m²)
                "ext_irradiance": spaces.Box(low=0, high=1500, shape=(1,), dtype=np.float32),
                # Time features (sin/cos encoding of hour and day)
                "datetime": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                # Occupant count
                "occupant_count": spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
                # Average workplane illuminance (lux)
                "avg_wpi": spaces.Box(low=0, high=5000, shape=(1,), dtype=np.float32),
            })
        if 'custom_action_space' in locals():
            self.action_space = custom_action_space
            print("Using custom action space")
        else:
            # Define action space
            # Format: [facade_state, cooling_setpoint, lighting_power]
            self.action_space = spaces.Box(
                # Low bounds for each action component
                low=np.array([0, 23, 0], dtype=np.float32),
                # High bounds for each action component
                high=np.array([1, 28, 1], dtype=np.float32),
                dtype=np.float32
            )


    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed) # Not used here but will initialize the random number generate: For example self.np_random.integers(0, self.size, size=2, dtype=int)
        
        # Reset the simulation
        self.simulation.reset()
        
        # Get initial observation (without taking an action)
        self.raw_next_obs = self.simulation.steps()
        
        # Process observation
        self.observation = self._process_observation(self.raw_next_obs)
        self.raw_observation = self.raw_next_obs
        
        # Return initial observation and info
        self.info={'number_of_timesteps_per_hour': self.simulation.number_of_timesteps_per_hour}

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
        sim_action = self.process_action(action)
        
        # Apply action and get next observation
        self.raw_next_obs = self.simulation.steps(sim_action)
        
        # Process observation from simulation
        next_observation = self._process_observation(self.raw_next_obs)
        
        # Calculate reward state, action, next_state
        reward = self.reward_function(self.observation, action, next_observation, self.info)

        # Log data
        if self.logging:
            self.log_data(self.raw_observation, action, self.raw_next_obs, self.info, reward)

        # Update the current observation
        self.observation = next_observation
        self.raw_observation = self.raw_next_obs
        
        if self.render_mode == "raw_obs":
            print(self.raw_next_obs)
        elif self.render_mode == "raw_time":
            print(self.raw_next_obs['datetime'])

        # Check if episode is done
        truncated = False  # Only truncated if simulation signals completion
        # Check if simulation is finished
        if self.raw_next_obs.get('simulation_finished', False):
            truncated = True
        
        terminated = False  # Not used in this context

        # Additional info
        info = {}

        return next_observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources when environment is no longer needed."""
        if hasattr(self, 'simulation'):
            self.simulation.shutdown()
    
    def render(self):
        """
        Render the environment.
        """
        if not hasattr(self, 'observation'):
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

            # General handling for any CFS state keys
            elif "cfs_state" in key and key in raw_obs:
                # Map CFS state string to float value
                cfs_value = 0.0  # default value
                if raw_obs[key] == "ec01":
                    cfs_value = 0.125  # CLEAR - middle of first quartile
                elif raw_obs[key] == "ec06":
                    cfs_value = 0.375  # LIGHT TINT - middle of second quartile
                elif raw_obs[key] == "ec18":
                    cfs_value = 0.625  # MEDIUM TINT - middle of third quartile
                elif raw_obs[key] == "ec60":
                    cfs_value = 0.875  # DARK - middle of fourth quartile
                # Store as a single-element array
                processed_obs[key] = np.array([cfs_value], dtype=np.float32)

            elif key in raw_obs:
                # Process keys that exist directly in raw_obs
                processed_obs[key] = np.array([raw_obs[key]], dtype=np.float32)
            else:
                # Default for keys not found in raw_obs
                processed_obs[key] = np.zeros(self.observation_space.spaces[key].shape, 
                                             dtype=self.observation_space.spaces[key].dtype)
        
        return processed_obs
    
    def _process_action(self, action):
        """
        Convert gym action to simulation action format.
        
        Args:
            action: Action from gym environment
            
        Returns:
            dict: Action in simulation format
        """
        # Extract action components
        facade_state_value = float(action[0])
        cooling_setpoint = float(action[1])
        lighting_power_frac = float(action[2])
        
        # Map continuous facade state value (0-1) to discrete states (0-3)
        if facade_state_value < 0.25:
            cfs_state = "ec01" # CLEAR
        elif facade_state_value < 0.5:
            cfs_state = "ec06" # LIGHT
        elif facade_state_value < 0.75:
            cfs_state = "ec18" # MEDIUM
        else:
            cfs_state = "ec60" # DARK
        
        # Calculate absolute lighting power (max is 1200W)
        lighting_power = lighting_power_frac * 1200.0
        
        # # Get current observation for datetime and other required fields, getattr using if variable could not exist in the first step
        # current_obs = getattr(self, 'observation', None)
        
        # # If no current observation (e.g., first step), use placeholder values
        # if current_obs is None:
        #     return None
        
        # Return action in format expected by simulation
        return {
            'cfs_state': cfs_state,
            'clg_setpoint': cooling_setpoint,
            'lighting_power': lighting_power
            # ,
            # # Pass through original data for logging
            # 'ext_irradiance': current_obs['ext_irradiance'],
            # 'datetime': current_obs['datetime'],
            # 'occupant_count': current_obs['occupant_count'],
            # 'avg_wpi': current_obs['avg_wpi']
        }

    def _calculate_reward(self, observation, action):
        """
        Calculate a simple reward based on visual comfort and energy usage.
        
        Args:
            observation: Current observation (matches observation_space)
            action: Action taken (facade_state, cooling_setpoint, lighting_power)
            
        Returns:
            float: Calculated reward
        """
        # Extract relevant values from observation
        occupant_count = observation['occupant_count'][0]
        avg_wpi = observation['avg_wpi'][0]
        
        # Extract action components
        facade_state = action[0]  # 0-1 value for facade transparency
        cooling_setpoint = action[1]  # temperature setpoint
        lighting_power = action[2]  # 0-1 fraction of max lighting power
        
        # Simple reward components
        
        # 1. Visual comfort - reward for maintaining illuminance between 300-500 lux when occupied
        visual_comfort = 0
        if occupant_count > 0:
            if 300 <= avg_wpi <= 500:
                visual_comfort = 1.0
            else:
                visual_comfort = max(0, 1.0 - min(abs(avg_wpi - 400) / 200, 1.0))
        
        # 2. Energy efficiency - penalize energy usage
        # Simple penalty based on lighting power and low cooling setpoints
        energy_penalty = lighting_power * 0.5 + max(0, (26 - cooling_setpoint) / 3) * 0.5
        
        # Combine rewards (visual comfort is positive, energy penalty is negative)
        total_reward = visual_comfort - energy_penalty
        
        return total_reward
    

    def log_data(self, observation, action, next_observation, info, reward):
        """
        Log environment data to a CSV file.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            info: Additional information
            reward: Reward received
        """
        
        log_path = os.path.join(self.output_dir, "environment_log.csv")
        file_exists = os.path.isfile(log_path)
        
        # Initialize file if not opened yet
        if not hasattr(self, 'log_file'):
            self.log_file = open(log_path, 'a', newline='')
            self.csv_writer = csv.writer(self.log_file)
        
        # If file doesn't exist, create it and write headers
        if not file_exists:         
            # Add observation keys (excluding datetime which is our first column)
            obs_keys = [f"obs_{k}" for k in observation.keys()]
            
            # Add action components
            action_keys = [f"action_{i}" for i in range(len(action))]
            
            # Add next_observation keys
            next_obs_keys = [f"next_obs_{k}" for k in next_observation.keys()]
            
            # Add info keys
            info_keys = [f"info_{k}" for k in info.keys()]
            
            # Add reward
            reward_key = ["reward"]
            
            # Write header row
            self.csv_writer.writerow(obs_keys + action_keys + next_obs_keys + info_keys + reward_key)
        
        # Prepare data row
        row = []
        
        # Add observation values
        for k in observation.keys():
            if isinstance(observation[k], np.ndarray):
                # Flatten array values and add them individually
                for val in observation[k].flatten():
                    row.append(float(val))
            else:
                row.append(observation[k])
        
        # Add action values
        for val in action:
            row.append(float(val))
        
        # Add next_observation values
        for k in next_observation.keys():
            if isinstance(next_observation[k], np.ndarray):
                # Flatten array values and add them individually
                for val in next_observation[k].flatten():
                    row.append(float(val))
            else:
                row.append(next_observation[k])
        
        # Add info values
        for k in info.keys():
            if isinstance(info[k], np.ndarray):
                # Flatten array values and add them individually
                for val in info[k].flatten():
                    row.append(float(val))
            else:
                row.append(info[k])
        
        # Add reward
        row.append(float(reward))
        
        # Write the data row
        self.csv_writer.writerow(row)
        self.log_file.flush()  # Ensure data is written immediately