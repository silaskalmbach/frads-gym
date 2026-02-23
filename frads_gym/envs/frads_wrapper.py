"""
FRADS Wrapper for Building Control Applications

This script provides a wrapper for the FRADS (Framework for Radiance And Daylighting Simulation) 
package that allows external control of building simulations at each timestep.

The wrapper reverses the traditional EnergyPlus control flow, making external programs the master
controller rather than EnergyPlus. This enables:
- Step-by-step control of the building simulation
- Integration with reinforcement learning algorithms
- Implementation of custom control strategies
- Co-simulation with other tools and models

Author: Silas Kalmbach
Date: April 2025
License: MIT
"""

# Import necessary libraries
import frads as fr
from pyenergyplus.dataset import ref_models, weather_files
import os
from epmodel import epmodel as epm
import threading
import queue
import time
import datetime
import shutil
from pathlib import Path
import json

# settings = fr.Settings()
# settings.num_processors = 4

class SimulationContext:
    """
    Singleton class that maintains global access to the active simulation.
    This allows the controller function to access simulation resources.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SimulationContext()
        return cls._instance
    
    def __init__(self):
        self.simulation = None  # Reference to the active EnergyPlusSimulation

# Global context variable
context = SimulationContext.get_instance()

class FradsSimulation:
    """
    Main class for managing the EnergyPlus/FRADS simulation.
    
    This class handles the initialization, execution, and communication
    with the EnergyPlus/FRADS simulation. It provides methods for:
    - Setting up the simulation environment
    - Running the simulation in a separate thread
    - Communicating with external controllers
    - Managing simulation state and resources
    """
    def __init__(self, output_dir=None, config_file=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, number_of_timesteps_per_hour=1, enable_radiance=True):
        """
        Initialize the EnergyPlus simulation.
        
        Args:
            output_dir (str, optional): Directory where simulation will store output files.
                                        If None, uses the directory where the script is located.
            config_file (str or dict, optional): Path to the configuration file OR a dictionary containing configuration. 
                                                 If None, uses "epsetup_config.json" in the script directory.
            run_annual (bool): If True, run the simulation in annual mode.
            cleanup (bool): If True, remove existing output directory before starting the simulation.
            run_period (dict, optional): Dictionary specifying the run period for the simulation.
                                        Example: {"begin_year": 2002, "begin_month": 1, "begin_day": 1, "end_year": 2002, "end_month": 12, "end_day": 31}
            treat_weather_as_actual (bool): If True, treat weather data as actual.
            weather_files_path (str, optional): Path to the weather files directory. ["weather_file1.epw", "weather_file2.epw"]
            number_of_timesteps_per_hour (int): Number of timesteps per hour for the simulation.
        """
        # simulation settings
        self.run_annual = run_annual
        self.run_period=run_period
        self.treat_weather_as_actual=treat_weather_as_actual
        self.number_of_timesteps_per_hour=number_of_timesteps_per_hour
        self.action_values = None 
        self.weather_files_path=weather_files_path
        self.current_weather_idx = 0
        self.enable_radiance = enable_radiance
        self.config_file = config_file
        self.simulation_finished = False

        loaded_config = {}

        # Determine the input directory and load config
        if config_file is not None:
            if isinstance(config_file, str):
                # It is a path
                config_path = os.path.abspath(config_file)
                self.input_dir = os.path.dirname(config_path)
                
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Inject origin path and file path into the json object
                loaded_config['config_dir'] = self.input_dir
                loaded_config['config_file_path'] = config_path

            elif isinstance(config_file, dict):
                # It is already a json object (dict)
                loaded_config = config_file
                # Try to retrieve input_dir from injected key
                self.input_dir = loaded_config.get('config_dir')
                
                if self.input_dir is None:
                    # Fallback if not present
                    self.input_dir = os.path.dirname(os.path.abspath(__file__))
            else:
                 raise ValueError("config_file must be a string path or a dictionary key-value map.")
        else:
            # Use the script directory as input directory
            self.input_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set the working directory for simulation outputs
        if output_dir is None:
            self.output_dir = self.input_dir
        else:
            # Create the directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir

        # Filter out inactive entries
        self.epsetup_config = {k: v for k, v in loaded_config.items() if isinstance(v, dict) and v.get("active", True)}
        print(f"Loaded {len(loaded_config)} config entries, kept {len(self.epsetup_config)} active entries")
        
        os.chdir(self.output_dir) # change chdir to output dir
        print(f"Output directory: {self.output_dir}")
        self.output_path_ep = os.path.join(self.output_dir, "eplusout")
        print(f"Output path for EnergyPlus: {self.output_path_ep}")

        if cleanup:
            self._cleanup_output(folders=["eplusout", "Temp", "Octrees", "Matrices"])

        # Create the output directory for EnergyPlus
        os.makedirs(self.output_path_ep, exist_ok=True)
        print(f"Output directory created: {self.output_path_ep}")
            
        # Threading components for communication with external processes
        self.obs_data_queue = queue.Queue(maxsize=1)   # Queue for observation data
        self.action_data_queue = queue.Queue(maxsize=1) # Queue for action data
        self.next_step_event = threading.Event()       # Synchronization event
        self.shutdown_event = threading.Event()        # Shutdown signal
        self.simulation_lock = threading.RLock()        # Reentrant lock for thread-safe operations

        # Initialize the simulation environment
        self._model_setup()
        self._simulation_config()

    def _model_setup(self):
        """
        Configure the EnergyPlus model based on JSON configuration.
        Supports both absolute and relative model paths as well as model types.
        """
        # Look for model_setup section in epsetup_config
        model_setup = self.epsetup_config.get("model_setup", {})
        
        # Try to get model path or type
        model_path = model_setup.get("model_path")
        model_type = model_setup.get("model_type")
        
        # Load model based on available configuration
        if model_path:
            # Check if path is absolute or relative
            if os.path.isabs(model_path):
                full_model_path = model_path
            else:
                # Resolve relative path using input_dir
                full_model_path = os.path.join(self.input_dir, model_path)
            
            # Check if the model file exists
            if os.path.exists(full_model_path):
                print(f"Loading model from path: {full_model_path}")
                self.epmodel = fr.load_energyplus_model(full_model_path)
            else:
                print(f"Warning: Model path not found: {full_model_path}")
                if model_type and model_type in ref_models:
                    print(f"Falling back to model type: {model_type}")
                    self.epmodel = fr.load_energyplus_model(ref_models[model_type])
                else:
                    available_models = list(ref_models.keys())
                    raise ValueError(f"Invalid model_path and no valid model_type. Available model types: {available_models}")
        elif model_type and model_type in ref_models:
            print(f"Loading model by type: {model_type}")
            self.epmodel = fr.load_energyplus_model(ref_models[model_type])
        else:
            # If neither path nor type works, raise error
            available_models = list(ref_models.keys())
            raise ValueError(f"No valid model_path or model_type provided. Available model types: {available_models}")
        
        # Configure lighting systems
        for lighting in model_setup.get("lighting_systems", []):
            self.epmodel.add_lighting(
                zone=lighting["zone"],
                lighting_level=lighting["level"],
                replace=lighting.get("replace", True)
            )
            print(f"Added lighting to zone {lighting['zone']}")
            
        # Configure glazing systems
        for glazing in model_setup.get("glazing_systems", []):
            # Check if path is absolute or relative
            if os.path.isabs(glazing["file"]):
                gs_path = glazing["file"]
            else:
                gs_path = os.path.join(self.input_dir, glazing["file"])
                
            if not os.path.exists(gs_path):
                print(f"Warning: Glazing system file not found: {gs_path}")
                continue
                
            # Load and add glazing system
            # gs = fr.GlazingSystem.from_json(gs_path) # frads version V1
            gs = fr.load_glazing_system(gs_path) # frads version V2
            self.epmodel.add_glazing_system(gs)

            print(f"Added glazing system from {gs_path}")

    def _simulation_config(self):
        """Configure simulation parameters using the first available run period."""
        
        # Get the first run period key dynamically
        run_period_keys = list(self.epmodel.run_period.keys())
        if not run_period_keys:
            raise ValueError("No run periods found in the EnergyPlus model.")
        
        # Use the first run period (instead of hardcoding 'annual')
        first_run_period_key = run_period_keys[0]
        print(f"Using run period with name: {first_run_period_key}")
        
        if self.run_period is not None:
            # Set the run period if provided
            if 'begin_year' in self.run_period:
                self.epmodel.run_period[first_run_period_key].begin_year = self.run_period['begin_year']
            if 'begin_month' in self.run_period:
                self.epmodel.run_period[first_run_period_key].begin_month = self.run_period['begin_month']
            if 'begin_day' in self.run_period:
                self.epmodel.run_period[first_run_period_key].begin_day_of_month = self.run_period['begin_day']
            if 'end_year' in self.run_period:
                self.epmodel.run_period[first_run_period_key].end_year = self.run_period['end_year']
            if 'end_month' in self.run_period:
                self.epmodel.run_period[first_run_period_key].end_month = self.run_period['end_month']
            if 'end_day' in self.run_period:
                self.epmodel.run_period[first_run_period_key].end_day_of_month = self.run_period['end_day']
    
        if self.treat_weather_as_actual:
            # Set the weather treatment if specified
            self.epmodel.run_period[first_run_period_key].treat_weather_as_actual = epm.EPBoolean.yes
        else:
            # Set the weather treatment to actual
            self.epmodel.run_period[first_run_period_key].treat_weather_as_actual = epm.EPBoolean.no

        # Set simulation timestep and control
        self.epmodel.timestep['Timestep 1'].number_of_timesteps_per_hour = self.number_of_timesteps_per_hour
        self.epmodel.simulation_control['SimulationControl 1'].run_simulation_for_weather_file_run_periods = epm.EPBoolean.yes
        self.epmodel.simulation_control['SimulationControl 1'].run_simulation_for_sizing_periods = epm.EPBoolean.no

        # Set annual period if requested
        if self.run_annual:
            self.epmodel.run_period[first_run_period_key].begin_month = 1
            self.epmodel.run_period[first_run_period_key].begin_day_of_month = 1
            self.epmodel.run_period[first_run_period_key].end_month = 12
            self.epmodel.run_period[first_run_period_key].end_day_of_month = 31
            
        # Ensure years are set
        if not self.epmodel.run_period[first_run_period_key].begin_year or not self.epmodel.run_period[first_run_period_key].end_year:
            self.epmodel.run_period[first_run_period_key].begin_year = 2002
            self.epmodel.run_period[first_run_period_key].end_year = 2002
            
        # Calculate total timesteps
        start_date = datetime.datetime(
            int(self.epmodel.run_period[first_run_period_key].begin_year), 
            int(self.epmodel.run_period[first_run_period_key].begin_month), 
            int(self.epmodel.run_period[first_run_period_key].begin_day_of_month), 
            0, 0, 0
        )
        end_date = datetime.datetime(
            int(self.epmodel.run_period[first_run_period_key].end_year), 
            int(self.epmodel.run_period[first_run_period_key].end_month), 
            int(self.epmodel.run_period[first_run_period_key].end_day_of_month), 
            23, 59, 59
        )
        
        # Calculate the sum of timesteps
        self.sum_timesteps = int(self.epmodel.timestep['Timestep 1'].number_of_timesteps_per_hour * 
                                 ((end_date - start_date).total_seconds()/3600))
        print(f"Total number of timesteps: {self.sum_timesteps}")

    def _wait_for_simulation_to_finish(self, timeout=300):
        """
        Wait until the simulation signals completion.

        This method blocks until the simulation thread signals that it has
        completed execution, either normally or due to an error.

        Args:
            timeout: Maximum seconds to wait before giving up.
        """
        start = time.time()
        # Signal the simulation to shut down and wake the controller
        self.shutdown_event.set()
        self.next_step_event.set()

        while not self.simulation_finished:
            elapsed = time.time() - start
            if elapsed > timeout:
                print(f"Warning: Timed out waiting for simulation to finish after {timeout}s. Forcing cleanup.")
                self.simulation_finished = True
                break
            try:
                obs_data = self.obs_data_queue.get(timeout=1.0)
                if isinstance(obs_data, dict) and obs_data.get('simulation_finished', False):
                    self.simulation_finished = True
            except queue.Empty:
                pass

        # Drain any remaining items from queues
        self._drain_queues()
        print("Simulation appears to be finished.")

    def _drain_queues(self):
        """Drain all pending items from communication queues."""
        for q in (self.obs_data_queue, self.action_data_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _register_controller(self):
        """
        Register the controller with EnergyPlus without triggering AST analysis errors.
        """
        # Register the simplified wrapper instead of the complex controller
        self.epsetup.set_callback(
        "callback_begin_system_timestep_before_predictor",
        simplified_controller_wrapper
        )

    def reset(self):
        """
        Reset and initialize the simulation.

        Cleanly shuts down any running EnergyPlus process, resets all
        synchronization primitives, and starts a fresh simulation thread.
        """
        # If a simulation is already running, shut it down
        if hasattr(self, 'epsetup'):
            self._wait_for_simulation_to_finish()
            try:
                self.epsetup.close()
            except Exception as e:
                print(f"Warning: Error closing previous EnergyPlus session: {e}")

        # Reset all synchronization primitives for the new episode
        self.shutdown_event.clear()
        self.next_step_event.clear()
        self.simulation_finished = False
        self._drain_queues()

        # Select the appropriate weather file
        if hasattr(self, 'weather_files_path') and self.weather_files_path and len(self.weather_files_path) > 0:
            current_weather = self.weather_files_path[self.current_weather_idx]
            self.current_weather_idx = (self.current_weather_idx + 1) % len(self.weather_files_path)
            print(f"Using weather file: {current_weather} (index {self.current_weather_idx-1})")
        else:
            current_weather = weather_files["usa_ca_san_francisco"]
            print("Using default San Francisco weather file")

        # Initialize EnergyPlus Simulation Setup with Radiance enabled
        self.epsetup = fr.EnergyPlusSetup(
            self.epmodel,
            current_weather,
            enable_radiance=self.enable_radiance
        )

        # Set the simulation instance in the global context singleton
        context.simulation = self

        # Register the controller callback for each timestep
        self._register_controller()

        # Start the simulation in a background thread
        simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        simulation_thread.start()

    def _run_simulation(self):
        """
        Run the EnergyPlus simulation in a separate thread.
        
        This method is executed in a background thread and runs the
        actual EnergyPlus simulation. It notifies when the simulation
        is complete.
        """
        # output_prefix = os.path.join(self.output_path_ep, fr.utils.random_string(5))
        output_prefix = os.path.join(self.output_path_ep, "eplus")
        print(f"Running EnergyPlus simulation with output prefix: {output_prefix}")
        print(self.output_path_ep)

        try:
            self._register_variables()
            self.epsetup.run(
                    annual=self.run_annual, 
                    design_day=False,
                    output_directory=self.output_path_ep,
                    output_prefix=output_prefix)
        finally:
            print("EnergyPlus simulation completed.")
            # Signal that the simulation has ended
            self.obs_data_queue.put({'simulation_finished': True})
            
    def steps(self, action_data=None):
        """
        Interface for external control of the simulation.

        This method advances the simulation by one timestep, applying
        the provided control actions and returning the new state.

        Args:
            action_data (dict, optional): Control actions to apply to the simulation.
                                         If None, no actions are taken.

        Returns:
            dict: Observation data from the simulation after the step.
        """
        if self.simulation_finished:
            return {'simulation_finished': True}

        try:
            # Send action data to controller
            if action_data:
                self.action_data_queue.put(action_data)

            # Signal controller to continue
            self.next_step_event.set()

            # Wait for data from controller
            self.obs_data = self.obs_data_queue.get(timeout=120)

            # Check if the simulation has ended
            if isinstance(self.obs_data, dict) and self.obs_data.get('simulation_finished', False):
                self.simulation_finished = True
                return {'simulation_finished': True}

            return self.obs_data
        except queue.Empty:
            print("Warning: No observation data received within timeout (120s).")
            return {'error': 'No step data available (timeout)'}

    def shutdown(self):
        """
        Properly terminate the simulation and release resources.

        This method should always be called when ending the simulation
        to ensure all resources are properly released and threads are
        terminated.
        """
        print("Initiating simulation shutdown...")

        # Wait for the simulation to finish (signals shutdown internally)
        self._wait_for_simulation_to_finish()

        # Properly close EnergyPlus simulation
        if hasattr(self, 'epsetup'):
            try:
                self.epsetup.close()
                print("EnergyPlus resources released.")
            except Exception as e:
                print(f"Error closing EnergyPlus: {e}")

        print("Simulation shutdown completed successfully.")

    def _cleanup_output(self, folders=None):
        """
        Clean up the simulation output directory.
        
        This method removes the specified output directory and all its contents.
        It should be called when the simulation is no longer needed.
        """
        for folder in folders:
            folder_path = os.path.join(self.output_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Existing output directory removed: {folder_path}")

    def _register_variables(self):
        """
        Register variables with EnergyPlus and create observation/action spaces.
        """
        # Register each variable with get_variable_value 
        for var_id, var_info in self.epsetup_config.items():
            if "get_variable_value" in var_info:
                var_name = var_info["get_variable_value"]["name"]
                var_key = var_info["get_variable_value"]["key"]
                self.epsetup.request_variable(var_name, var_key)
                print(f"Registered variable: {var_name} for {var_key}")


def controller(state):
    """
    Controller function called at each EnergyPlus timestep.
    Uses the epsetup_config to dynamically collect observations.
    """
    # Do NOT CHANGE the following lines
    simulation = context.simulation
    
    # API readiness checks
    if not simulation.epsetup.api.exchange.api_data_fully_ready(state):
        return
    
    if simulation.epsetup.api.exchange.warmup_flag(state):
        return
    
    # Check if shutdown signal has been set
    if simulation.shutdown_event.is_set():
        return

    ###################################
    # Initialize observation data with datetime
    obs_data = {'datetime': simulation.epsetup.get_datetime()}
    
    # 1. First process all get_variable_value entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "get_variable_value" in var_info:
            var_name = var_info["get_variable_value"]["name"]
            var_key = var_info["get_variable_value"]["key"]
            obs_data[var_id] = simulation.epsetup.get_variable_value(
                name=var_name,
                key=var_key
            )
    
    # 2. Process all get_cfs_state entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "get_cfs_state" in var_info:
            window_key = var_info["get_cfs_state"]["key"]
            obs_data[var_id] = simulation.epsetup.get_cfs_state(window_key)
    
    # 3. Process all calculate_wpi entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "calculate_wpi" in var_info:
            zone = var_info["calculate_wpi"]["zone"]
            cfs_names = var_info["calculate_wpi"]["cfs_name"]
            
            # Create cfs_name dictionary mapping window names to their current states
            cfs_dict = {}
            for window in cfs_names:
                cfs_dict[window] = simulation.epsetup.get_cfs_state(window)
            
            # Calculate WPI
            wpi_result = simulation.epsetup.calculate_wpi(zone=zone, cfs_name=cfs_dict)
            
            # Apply post-processing if specified
            if "post_processing" in var_info and var_info["post_processing"] == "mean":
                obs_data[var_id] = wpi_result.mean()
            else:
                obs_data[var_id] = wpi_result
    
    # 4. Process all calculate_edgps entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "calculate_edgps" in var_info:
            zone = var_info["calculate_edgps"]["zone"]
            cfs_names = var_info["calculate_edgps"]["cfs_name"]
            
            # Create cfs_name dictionary mapping window names to their current states
            cfs_dict = {}
            for window in cfs_names:
                cfs_dict[window] = simulation.epsetup.get_cfs_state(window)
            
            # Calculate eDGPS
            obs_data[var_id] = simulation.epsetup.calculate_edgps(
                zone=zone, 
                cfs_name=cfs_dict
            )
    
    # 5. Process lighting power calculations for entries that depend on other values
    for var_id, var_info in simulation.epsetup_config.items():
        if "lighting_power_calculation" in var_info and "depends_on" in var_info:
            depends_on_key = var_info["depends_on"]
            
            # Check if the dependency is available in obs_data
            if depends_on_key in obs_data:
                # Get zone from the dependent key's configuration
                zone = simulation.epsetup_config[depends_on_key]["calculate_wpi"]["zone"]
                lux_threshold = var_info["lighting_power_calculation"]["lux_threshold"]
                
                # Determine power density based on configuration
                lighting_calc = var_info["lighting_power_calculation"]
                
                # Check if area-based calculation is specified
                if "area_m2" in lighting_calc and "power_density_m2" in lighting_calc:
                    # Calculate total power from power density per m² and area
                    power_density_m2 = lighting_calc["power_density_m2"]
                    area_m2 = lighting_calc["area_m2"]
                    power_density = power_density_m2 * area_m2
                    # print(f"Using area-based calculation for {var_id}: {power_density_m2} W/m² × {area_m2} m² = {power_density} W")
                elif "power_density" in lighting_calc:
                    # Use direct power density value (backwards compatibility)
                    power_density = lighting_calc["power_density"]
                else:
                    # print(f"Warning: Neither 'power_density' nor 'power_density_m2' + 'area_m2' specified for {var_id}")
                    obs_data[var_id] = 0.0
                    continue
                
                # Calculate lighting power based on the dependency value (WPI)
                wpi_value = obs_data[depends_on_key]
                lighting_power = (1 - min(wpi_value / lux_threshold, 1)) * power_density
                
                # Convert to energy (Joules)
                lighting_energy = lighting_power * 3600 / simulation.number_of_timesteps_per_hour
                obs_data[var_id] = lighting_energy
                
                # Actuate the lighting power in the simulation
                zone_name = zone.upper()
                simulation.epsetup.actuate_lighting_power(light=zone_name, value=lighting_power)
            else:
                print(f"Warning: Dependency '{depends_on_key}' not found for '{var_id}'")
                obs_data[var_id] = 0.0
    
    # 6. Process all divider entries BEFORE calculate_energy_sum
    for var_id, var_info in simulation.epsetup_config.items():
        if "divider" in var_info and var_id in obs_data:
            divider_value = var_info["divider"]
            try:
                obs_data[var_id] = float(obs_data[var_id]) / float(divider_value)
                # print(f"Applied divider {divider_value} to {var_id}: {obs_data[var_id]}")
            except (ValueError, TypeError, ZeroDivisionError) as e:
                print(f"Warning: Could not apply divider to {var_id}: {e}")
    
    # 7. Process all calculate_energy_sum entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "calculate_energy_sum" in var_info:
            # Reads the updated keys_to_sum from the JSON config
            keys_to_sum = var_info["calculate_energy_sum"].get("keys_to_sum", [])
            total_energy = 0
            for key in keys_to_sum:
                value = obs_data.get(key)
                if value is not None:
                    try:
                        total_energy += float(value)
                    except (ValueError, TypeError):
                         print(f"Warning: Value for key '{key}' ({value}) is not numeric. Skipping in energy sum for '{var_id}'.")
                else:
                    print(f"Warning: Key '{key}' not found in observation data for energy sum '{var_id}'. Assuming 0.")
            obs_data[var_id] = total_energy

    ###################################
    # Do NOT CHANGE the following lines
    # Put data into the queue for external controller
    simulation.obs_data_queue.put(obs_data)
    
    # Wait for signal to continue
    simulation.next_step_event.wait()
    simulation.next_step_event.clear()

    # Process action data from external controller
    action_values = None
    if not simulation.action_data_queue.empty():
        action_values = simulation.action_data_queue.get()
        if action_values is None:
            return
    else:
        return
    
    ###################################
    # Apply actions based on configuration
    for var_id, var_info in simulation.epsetup_config.items():
        if "actuate_cfs_state" in var_info:
            window_key = var_info["actuate_cfs_state"]["key"]
            action_key = var_id
            
            # Only actuate if this window has an action value
            if action_key in action_values:
                simulation.epsetup.actuate_cfs_state(
                    window=window_key,
                    cfs_state=action_values[action_key],
                )

# Create a simple wrapper function that FRADS can analyze correctly
def simplified_controller_wrapper(state):
    """Simple wrapper around the real controller"""
    return controller(state)


def random_action_values(simulation_data):
    """
    Generate random action values that match the expected format for the simulation.
    
    Args:
        simulation_data: Current observation data from simulation
        
    Returns:
        dict: Random action values in the format expected by the simulation
    """
    import random
    
    # Get the simulation instance from context
    simulation = context.simulation
    
    # Generate random continuous values between 0-1 for each window
    facade_state_1 = random.random()  # South window
    facade_state_2 = random.random()  # East window
    facade_state_3 = random.random()  # North window
    facade_state_4 = random.random()  # West window
    
    # Return action in format expected by simulation
    return {
        'cfs_state_1': _map_value_to_facade_state(simulation, facade_state_1),  # South window
        'cfs_state_2': _map_value_to_facade_state(simulation, facade_state_2),  # East window
        'cfs_state_3': _map_value_to_facade_state(simulation, facade_state_3),  # North window
        'cfs_state_4': _map_value_to_facade_state(simulation, facade_state_4),  # West window
    }

def _map_value_to_facade_state(simulation, value, facade_mapping=None):
    """
    Maps a continuous value (0-1) to a discrete facade state based on config mapping.
    
    Args:
        simulation: The simulation instance containing configuration
        value (float): A value between 0 and 1 representing the facade state
        facade_mapping (dict, optional): The facade state mapping configuration.
                                        If None, uses simulation.epsetup_config["facade_state_mapping"]
    
    Returns:
        str: The facade state code (e.g., "ec01", "ec06", "ec18", "ec60")
    """
    # Get the facade mapping configuration
    if facade_mapping is None and hasattr(simulation, 'epsetup_config'):
        facade_mapping = simulation.epsetup_config.get("facade_state_mapping", {})
    
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

