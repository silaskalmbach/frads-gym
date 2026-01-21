"""
frads-gym Example: Basic Environment Test

This example demonstrates how to use the frads-gym Gymnasium environment
for building facade control simulation with EnergyPlus and Radiance.

Requirements:
- frads-gym installed (`pip install -e .` from the frads-gym directory)
- EnergyPlus installed and accessible
- frads installed

Usage:
    python frads-gym_test.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Import frads-gym - this registers the environment with Gymnasium
import frads_gym
from frads_gym.envs.frads_gym import FradsEnv

# Optional: For vectorized environments with stable-baselines3
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Note: stable-baselines3 not installed. Using basic environment.")


def simple_reward(state, action, next_state, info):
    """
    Simple reward function based on total energy demand.
    Lower energy = higher reward.
    """
    energy = info.get('raw_next_total_energy_demand_1', [0.0])[0]
    # Negative reward proportional to energy consumption
    return -energy


def run_basic_example():
    """Run a basic example with the FradsEnv environment."""
    
    # Get the path to example data
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    config_path = data_dir / "example_config.json"
    weather_file = data_dir / "3C_USA_CA_SAN_FRANCISCO.epw"
    
    # Create output directory for simulation results
    output_dir = script_dir / f"simulation_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load and prepare config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Inject paths for relative path resolution
    config_dict['config_dir'] = str(data_dir)
    config_dict['config_file_path'] = str(config_path)
    
    print(f"Config loaded from: {config_path}")
    print(f"Weather file: {weather_file}")
    print(f"Output directory: {output_dir}")
    
    # Create the environment
    env = FradsEnv(
        output_dir=str(output_dir),
        config_file=config_dict,
        weather_files_path=[str(weather_file)],
        run_annual=False,  # Set to True for full year simulation
        cleanup=True,
        reward_function=simple_reward,
        logging=True,
        number_of_timesteps_per_hour=1,
        enable_radiance=True  # Set to False to disable Radiance calculations
    )
    
    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run simulation loop
    print("\n--- Starting Simulation ---\n")
    obs, info = env.reset()
    
    total_energy = 0.0
    total_steps = 0
    
    # Run for a limited number of steps (adjust as needed)
    max_steps = 100
    
    for step in range(max_steps):
        # Sample a random action or use a fixed action
        # action = env.action_space.sample()  # Random action
        action = [0.5]  # Fixed action: medium tint
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track energy consumption
        energy = info.get('raw_next_total_energy_demand_1', [0.0])[0]
        total_energy += energy
        total_steps += 1
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}: Reward={reward:.4f}, Energy={energy:.0f} J")
        
        if terminated or truncated:
            print(f"\nSimulation finished at step {step}")
            break
    
    # Close the environment
    env.close()
    
    print(f"\n--- Simulation Complete ---")
    print(f"Total steps: {total_steps}")
    print(f"Total energy: {total_energy:.0f} J")
    print(f"Average energy per step: {total_energy / total_steps:.0f} J")


def run_vectorized_example():
    """Run example with vectorized environment (requires stable-baselines3)."""
    
    if not SB3_AVAILABLE:
        print("stable-baselines3 not available. Skipping vectorized example.")
        return
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    config_path = data_dir / "example_config.json"
    weather_file = data_dir / "3C_USA_CA_SAN_FRANCISCO.epw"
    output_dir = script_dir / f"simulation_vec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def make_env():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict['config_dir'] = str(data_dir)
        config_dict['config_file_path'] = str(config_path)
        
        return FradsEnv(
            output_dir=str(output_dir),
            config_file=config_dict,
            weather_files_path=[str(weather_file)],
            run_annual=False,
            cleanup=True,
            reward_function=simple_reward,
            logging=False,
            number_of_timesteps_per_hour=1
        )
    
    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env])
    vec_env = VecMonitor(vec_env)
    
    print("Vectorized environment created successfully!")
    
    obs = vec_env.reset()
    for _ in range(10):
        action = [[0.5]]
        obs, rewards, dones, infos = vec_env.step(action)
        print(f"Reward: {rewards[0]:.4f}")
    
    vec_env.close()
    print("Vectorized example complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("frads-gym Example: Building Facade Control Environment")
    print("=" * 60)
    
    # Run basic example
    run_basic_example()
    
    # Uncomment to run vectorized example (requires stable-baselines3)
    # run_vectorized_example()
