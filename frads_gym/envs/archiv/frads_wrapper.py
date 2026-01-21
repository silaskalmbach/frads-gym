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

class EnergyPlusSimulation:
    """
    Main class for managing the EnergyPlus/FRADS simulation.
    
    This class handles the initialization, execution, and communication
    with the EnergyPlus/FRADS simulation. It provides methods for:
    - Setting up the simulation environment
    - Running the simulation in a separate thread
    - Communicating with external controllers
    - Managing simulation state and resources
    """
    def __init__(self, output_dir=None, input_dir=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, number_of_timesteps_per_hour=1):
        """
        Initialize the EnergyPlus simulation.
        
        Args:
            output_dir (str, optional): Directory where simulation will store output files.
                                        If None, uses the directory where the script is located.
            input_dir (str, optional): Directory where the script is located. If None, uses the current directory.
            run_annual (bool): If True, run the simulation in annual mode.
            cleanup (bool): If True, remove existing output directory before starting the simulation.
            run_period (dict, optional): Dictionary specifying the run period for the simulation.
                                        Example: {"begin_year": 2000, "begin_month": 1, "begin_day": 1, "end_year": 2000, "end_month": 12, "end_day": 31}
            treat_weather_as_actual (bool): If True, treat weather data as actual.
            weather_files_path (str, optional): Path to the weather files directory. ["weather_file1.epw", "weather_file2.epw"]
            number_of_timesteps_per_hour (int): Number of timesteps per hour for the simulation.
        """
        # simulation settings
        self.run_annual = run_annual
        self.run_period=run_period
        self.treat_weather_as_actual=treat_weather_as_actual
        self.number_of_timesteps_per_hour=number_of_timesteps_per_hour
        self.action_values = None # for the controller function TODO:Help
        # Initialize weather file index
        self.weather_files_path=weather_files_path
        self.current_weather_idx = 0

        if input_dir is None:
            # Get the script directory (where input files are located)
            self.input_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            # Use the provided input directory
            self.input_dir = input_dir
          
        # Set the working directory for simulation outputs
        if output_dir is None:
            self.output_dir = self.input_dir
        else:
            # Create the directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir

        os.chdir(self.output_dir) # change chdir to output dir

        # # Apply the monkey patch to FRADS
        # patch_frads_paths(self.output_dir)
        # Path for output files of eplus
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
        Configure the EnergyPlus model.
        """
        if hasattr(self, 'custom_model_setup') and self.custom_model_setup:
            # Call custom model_setup function with self as argument
            self.custom_model_setup(self)
        else:
            # Default implementation
            model_path=ref_models["medium_office"]
            self.epmodel = fr.load_energyplus_model(model_path)

            # Add lighting systems to the EnergyPlus model
            self.epmodel.add_lighting(
                zone="Perimeter_bot_ZN_1",
                lighting_level=1200,  # Lighting level in watts
                replace=True
            )

            # Load the glazing systems from JSON files (from input_dir)
            gs_ec01 = fr.GlazingSystem.from_json(os.path.join(self.input_dir, "gs_ec01.json"))
            gs_ec06 = fr.GlazingSystem.from_json(os.path.join(self.input_dir, "gs_ec06.json"))
            gs_ec18 = fr.GlazingSystem.from_json(os.path.join(self.input_dir, "gs_ec18.json"))
            gs_ec60 = fr.GlazingSystem.from_json(os.path.join(self.input_dir, "gs_ec60.json"))

            # Add the glazing systems to the EnergyPlus model
            self.epmodel.add_glazing_system(gs_ec01)
            self.epmodel.add_glazing_system(gs_ec06)
            self.epmodel.add_glazing_system(gs_ec18)
            self.epmodel.add_glazing_system(gs_ec60)

            # Initialize or clear the states log file (in output_dir)
            self.states_file_path = os.path.join(self.output_dir, "states.csv")
            if os.path.exists(self.states_file_path):
                os.remove(self.states_file_path)

    def _simulation_config(self):
        """
        Load an EnergyPlus model from the specified path.
        
        Args:
            run_annual (bool): If True, run the simulation in annual mode.
            run_period (dict): Optional run period for the simulation. {"begin_year": 2000, "begin_month": 1, "begin_day": 1, "end_year": 2000, "end_month": 12, "end_day": 31}
        """

        if self.run_period is not None:
            # Set the run period if provided
            if 'begin_year' in self.run_period:
                self.epmodel.run_period['annual'].begin_year = self.run_period['begin_year']
            if 'begin_month' in self.run_period:
                self.epmodel.run_period['annual'].begin_month = self.run_period['begin_month']
            if 'begin_day' in self.run_period:
                self.epmodel.run_period['annual'].begin_day_of_month = self.run_period['begin_day']
            if 'end_year' in self.run_period:
                self.epmodel.run_period['annual'].end_year = self.run_period['end_year']
            if 'end_month' in self.run_period:
                self.epmodel.run_period['annual'].end_month = self.run_period['end_month']
            if 'end_day' in self.run_period:
                self.epmodel.run_period['annual'].end_day_of_month = self.run_period['end_day']
 
        if self.treat_weather_as_actual:
            # Set the weather treatment if specified
            self.epmodel.run_period['annual'].treat_weather_as_actual = epm.EPBoolean.yes
        else:
            # Set the weather treatment to actual
            self.epmodel.run_period['annual'].treat_weather_as_actual = epm.EPBoolean.no

        # Set simulation timestep and control
        self.epmodel.timestep['Timestep 1'].number_of_timesteps_per_hour = self.number_of_timesteps_per_hour # 12 timesteps per hour eqals 5 min timestep
        self.epmodel.simulation_control['SimulationControl 1'].run_simulation_for_weather_file_run_periods = epm.EPBoolean.yes
        self.epmodel.simulation_control['SimulationControl 1'].run_simulation_for_sizing_periods = epm.EPBoolean.no

        # # Calcualte the sum of hours based on the run_period information
        if self.run_annual:
            self.epmodel.run_period['annual'].begin_month = 1
            self.epmodel.run_period['annual'].begin_day_of_month = 1
            self.epmodel.run_period['annual'].end_month = 12
            self.epmodel.run_period['annual'].end_day_of_month = 31
        if self.epmodel.run_period['annual'].begin_year or self.epmodel.run_period['annual'].end_year is None:
            self.epmodel.run_period['annual'].begin_year, self.epmodel.run_period['annual'].end_year = 2000, 2000
        start_date = datetime.datetime(self.epmodel.run_period['annual'].begin_year, self.epmodel.run_period['annual'].begin_month, self.epmodel.run_period['annual'].begin_day_of_month, 0, 0, 0)
        end_date = datetime.datetime(self.epmodel.run_period['annual'].end_year, self.epmodel.run_period['annual'].end_month, self.epmodel.run_period['annual'].end_day_of_month, 23, 59, 59)
        # Calculate the sum of timesteps
        self.sum_timesteps = int(self.epmodel.timestep['Timestep 1'].number_of_timesteps_per_hour * ((end_date - start_date).total_seconds()/3600))
        print(f"Total number of timesteps: {self.sum_timesteps}")

    def _wait_for_simulation_to_finish(self):
        """
        Wait until the simulation signals completion.
        
        This method blocks until the simulation thread signals that it has
        completed execution, either normally or due to an error.
        """
        while self.simulation_finished == False:
            try:
                finished = self.obs_data_queue.get('simulation_finished', False)
                if finished:
                    self.simulation_finished = True
            except queue.Empty:
                finished = False
            time.sleep(0.05)        
        print("Simulation appears to be finished.")

    def _register_controller(self):
        """
        Register a callback function to be called at each timestep.
        
        Args:
            controller (callable): The controller function to be called.
        """
        # Register the controller callback for each timestep
        self.epsetup.set_callback(
            "callback_begin_system_timestep_before_predictor",
            controller
        )

    def reset(self):
        """
        Reset and initialize the simulation.
        """
        # If a simulation is already running, shut it down
        if hasattr(self, 'epsetup'):
            if hasattr(self, 'shutdown_event'):
                self.shutdown_event.set()  

            self._wait_for_simulation_to_finish()
            self.epsetup.close()
            # Reset the shutdown event
            self.shutdown_event.clear()

        # Select the appropriate weather file
        if hasattr(self, 'weather_files_path') and self.weather_files_path and len(self.weather_files_path) > 0:
            # Get the current weather file
            current_weather = self.weather_files_path[self.current_weather_idx]
            # Update the index for next time, wrapping around if needed
            self.current_weather_idx = (self.current_weather_idx + 1) % len(self.weather_files_path)
            print(f"Using weather file: {current_weather} (index {self.current_weather_idx-1})")
        else:
            # Use the default weather file if no list provided
            current_weather = weather_files["usa_ca_san_francisco"]
            print("Using default San Francisco weather file")

        self._cleanup_output(folders=["Octrees"])

        # Initialize EnergyPlus Simulation Setup with Radiance enabled
        self.epsetup = fr.EnergyPlusSetup(
            self.epmodel, 
            current_weather,
            enable_radiance=True
        )

        # Register Variables
        self._register_variables()

        # Set the simulation instance in the global context singleton
        context.simulation = self

        # Register the controller callback for each timestep
        self._register_controller()

        # Start the simulation in a background thread
        self.simulation_finished = False
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
        try:
            # Send action data to controller
            if action_data:
                self.action_data_queue.put(action_data)
            
            # Clear any old data from the queue
            while not self.obs_data_queue.empty():
                self.obs_data_queue.get()
                
            # Signal controller to continue
            self.next_step_event.set()
            
            # Wait for data from controller
            self.obs_data = self.obs_data_queue.get(timeout=10)
            
            # Check if the simulation has ended
            if self.obs_data.get('simulation_finished', False):
                self.simulation_finished = True
                return {'simulation_finished': True}

            return self.obs_data
        except queue.Empty:
            return {'error': 'No step data available (timeout)'}

    def shutdown(self):
        """
        Properly terminate the simulation and release resources.
        
        This method should always be called when ending the simulation
        to ensure all resources are properly released and threads are
        terminated.
        """
        print("Initiating simulation shutdown...")
        
        # Signal all threads to terminate
        self.shutdown_event.set()
        
        # Resolve any blocking wait() operations
        self.next_step_event.set()
        
        # Wait for the simulation to finish - at shutdown, this should be immediate
        self._wait_for_simulation_to_finish()

        # Properly close EnergyPlus simulation
        if hasattr(self, 'epsetup'):
            try:
                self.epsetup.close()
                print("EnergyPlus resources released.")
            except Exception as e:
                print(f"Error closing EnergyPlus: {e}")
        
        # Clear queues to avoid memory leaks
        try:
            while not self.obs_data_queue.empty():
                self.obs_data_queue.get_nowait()
            while not self.action_data_queue.empty():
                self.action_data_queue.get_nowait()
        except Exception:
            pass
        
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

def controller(state):
    """
    Controller function called at each EnergyPlus timestep.
    
    This function:
    1. Collects data from the current simulation state
    2. Passes it to the external controller via a queue
    3. Waits for control actions from the external controller
    4. Applies those actions to the EnergyPlus simulation
    
    Args:
        state: EnergyPlus state object provided by the callback
        
    Returns:
        None
    """
    simulation = context.simulation
    
    # API readiness checks
    if not simulation.epsetup.api.exchange.api_data_fully_ready(state):
        return
    
    if simulation.epsetup.api.exchange.warmup_flag(state):
        return
    
    # Check if shutdown signal has been set
    if simulation.shutdown_event.is_set():
        return

    # Collect all input data for calculation
    get_cfs_state_1 = simulation.epsetup.get_cfs_state("Perimeter_bot_ZN_1_Wall_South_Window")
    get_cfs_state_2 = simulation.epsetup.get_cfs_state("Perimeter_bot_ZN_2_Wall_East_Window")
    get_cfs_state_3 = simulation.epsetup.get_cfs_state("Perimeter_bot_ZN_3_Wall_North_Window")
    get_cfs_state_4 = simulation.epsetup.get_cfs_state("Perimeter_bot_ZN_4_Wall_West_Window")
    print(f"Current CFS state: {get_cfs_state_1, get_cfs_state_2, get_cfs_state_3, get_cfs_state_4}")

    obs_data = {
        # Measurements from EnergyPlus
        'ext_irradiance': simulation.epsetup.get_variable_value(
            name="Surface Outside Face Incident Solar Radiation Rate per Area",
            key="Perimeter_bot_ZN_1_Wall_South_Window",
        ),
        'datetime': simulation.epsetup.get_datetime(),
        'occupant_count': simulation.epsetup.get_variable_value(
            name="Zone People Occupant Count", key="PERIMETER_BOT_ZN_1"
        ),
        'avg_wpi_1': simulation.epsetup.calculate_wpi(
            zone="Perimeter_bot_ZN_1",
            cfs_name={"Perimeter_bot_ZN_1_Wall_South_Window": get_cfs_state_1},
            ).mean(),
        'avg_wpi_2': simulation.epsetup.calculate_wpi(
            zone="Perimeter_bot_ZN_2",
            cfs_name={"Perimeter_bot_ZN_2_Wall_East_Window": get_cfs_state_2},
            ).mean(),
        'avg_wpi_3': simulation.epsetup.calculate_wpi(
            zone="Perimeter_bot_ZN_3",
            cfs_name={"Perimeter_bot_ZN_3_Wall_North_Window": get_cfs_state_3},
            ).mean(),
        'avg_wpi_4': simulation.epsetup.calculate_wpi(
            zone="Perimeter_bot_ZN_4",
            cfs_name={"Perimeter_bot_ZN_4_Wall_West_Window": get_cfs_state_4},
            ).mean(),
    }

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
    
    print(obs_data['datetime'])

    # Update values in EnergyPlus
    simulation.epsetup.actuate_cfs_state(
        window="Perimeter_bot_ZN_1_Wall_South_Window",
        cfs_state=action_values['cfs_state'],
    )
    simulation.epsetup.actuate_cfs_state(
        window="Perimeter_bot_ZN_2_Wall_East_Window",
        cfs_state=action_values['cfs_state'],
    )
    simulation.epsetup.actuate_cfs_state(
        window="Perimeter_bot_ZN_3_Wall_North_Window",
        cfs_state=action_values['cfs_state'],
    )
    simulation.epsetup.actuate_cfs_state(
        window="Perimeter_bot_ZN_4_Wall_West_Window",
        cfs_state=action_values['cfs_state'],
    )

    simulation.epsetup.actuate_cooling_setpoint(
        zone="Perimeter_bot_ZN_1", 
        value=action_values['clg_setpoint']
    )
    
    simulation.epsetup.actuate_lighting_power(
        light="Perimeter_bot_ZN_1",
        value=action_values['lighting_power'],
    )
    
    # Write values to log file
    with open(simulation.states_file_path, "a") as f:
        if not os.path.exists(simulation.states_file_path) or os.path.getsize(simulation.states_file_path) == 0:
            f.write("datetime,ext_irradiance,cfs_state,clg_setpoint,occupant_count,avg_wpi,lighting_power\n")
        
        f.write(f"{obs_data['datetime']},{obs_data['ext_irradiance']}," + 
                f"{action_values['cfs_state']},{action_values['clg_setpoint']}," +
                f"{obs_data['occupant_count']},{obs_data['avg_wpi_1']}," +
                f"{action_values['lighting_power']}\n")

def calculate_action_values(simulation_data):
    """
    Calculate control values based on simulation data.
    
    This is an example control algorithm that determines:
    - Facade shading state based on external irradiance
    - Cooling setpoint based on time of day
    - Lighting power based on occupancy and daylight levels
    
    Args:
        simulation_data (dict): Dictionary containing sensor data from the simulation
            
    Returns:
        dict: Control actions to apply to the simulation
    """
    # Safety check for incomplete data
    if not simulation_data or any(key not in simulation_data for key in ['ext_irradiance', 'datetime', 'occupant_count', 'avg_wpi_1']):
        print("Incomplete simulation data received. Cannot calculate control values.")
        return None
    
    # Extract required input data
    ext_irradiance = simulation_data.get('ext_irradiance')
    datetime = simulation_data.get('datetime')
    occupant_count = simulation_data.get('occupant_count')
    avg_wpi = simulation_data.get('avg_wpi_1')
    
    # Additional safety check for None values
    if ext_irradiance is None or datetime is None or occupant_count is None or avg_wpi is None:
        print("One or more required simulation values are None. Cannot calculate control values.")
        return None
    
    # Calculate facade shading based on radiation intensity
    # Facade shading state control algorithm
    if ext_irradiance <= 300:
        ec = "60"  # Most tinted state
    elif ext_irradiance <= 400 and ext_irradiance > 300:
        ec = "18"  # Medium tint
    elif ext_irradiance <= 450 and ext_irradiance > 400:
        ec = "06"  # Light tint
    elif ext_irradiance > 450:
        ec = "01"  # Clear state
    cfs_state = f"ec{ec}"
    
    # Calculate cooling setpoint based on time of day
    if datetime.hour >= 16 and datetime.hour < 21:
        clg_setpoint = 25.56  # Evening setback
    elif datetime.hour >= 12 and datetime.hour < 16:
        clg_setpoint = 21.67  # Peak cooling hours
    else:
        clg_setpoint = 24.44  # Standard setpoint
    
    # Calculate lighting power based on occupancy and daylight
    if occupant_count > 0:
        lighting_power = (
            1 - min(avg_wpi / 500, 1)
        ) * 1200  # 1200W is the nominal lighting power density
    else:
        lighting_power = 0  # No occupants, lights off
    
    # Return all calculated values as a dictionary
    return {
        'cfs_state': cfs_state,
        'clg_setpoint': clg_setpoint,
        'lighting_power': lighting_power
    }

if __name__ == "__main__":
    # Example usage of the EnergyPlusSimulation class
    # epsim = EnergyPlusSimulation(output_dir="/home/simulation/Test", input_dir="/home/model/medium_office", run_annual=False, cleanup=True, run_period={"begin_year": 2000, "begin_month": 1, "begin_day": 1, "end_year": 2000, "end_month": 1, "end_day": 14}, treat_weather_as_actual=False, weather_files_path=[weather_files["usa_ca_san_francisco"], weather_files["usa_nv_las_vegas"]], number_of_timesteps_per_hour=1)

    epsim = EnergyPlusSimulation(output_dir="/home/simulation/Test", run_annual=False, cleanup=True, run_period={"begin_year": 2000, "begin_month": 1, "begin_day": 1, "end_year": 2000, "end_month": 1, "end_day": 3}, treat_weather_as_actual=False, weather_files_path=[weather_files["usa_ca_san_francisco"], weather_files["usa_nv_las_vegas"]], number_of_timesteps_per_hour=1)

    # Run one simulation episode
    epsim.reset()
    observation_data = epsim.steps(action_data=None)

    # Simulate for 150 timesteps (e.g., 150 hours)
    i = 0
    k = 0
    # for i in range(150):
    while True:
        # Calculate control actions based on current observation
        action_values = calculate_action_values(observation_data)
        # print(action_values)
        
        # Apply actions and get next observation
        observation_data = epsim.steps(action_values)

        # Check if simulation has finished
        if observation_data.get('simulation_finished', False):
            print("Simulation has finished. Exiting cleanly...")
            # break
            i += 1
            if i > k:
                break
            epsim.reset()
            observation_data = epsim.steps(action_data=None)

    # Clean up resources
    epsim.shutdown()