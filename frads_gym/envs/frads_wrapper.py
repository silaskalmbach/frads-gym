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
from frads.methods import SensorConfig, ViewConfig
import pyradiance as pr
from pyenergyplus.dataset import ref_models
import os
import sys
from pyenergyplus.model import model as epm
import threading
import queue
import time
import datetime
import shutil
from pathlib import Path
import json

import hashlib
import numpy as np

from frads_gym.cfs_thermal import CFSThermalModel, add_other_equipment
from frads_gym.five_phase_wiring import check_frads_capabilities


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sensor_digest(wf):
    """Pruefsumme der Sensoren eines Radiance-Workflows fuer feste Matrizen.

    ponytail: Positionen auf 0,1 mm gerundet (+0.0 gegen -0.0), damit Linux und Windows denselben
    Wert liefern; Aenderungen am Radiance-Modell ausser Sensoren und Gebaeudemodell (z. B. andere
    CFS-Fenstergruppen in der Config) erkennt die Pruefung nicht.
    """
    sensors = {
        name: (np.round(np.asarray(cfg.data, dtype=float), 4) + 0.0).tolist() if cfg.data
        else Path(cfg.file).read_text()
        for name, cfg in sorted(wf.config.model.sensors.items())
    }
    return hashlib.sha256(json.dumps(sensors, sort_keys=True).encode()).hexdigest()

def apply_custom_radiance_config(epsetup, epsetup_config: dict) -> set[str]:
    """Apply custom sensor/view positions from *epsetup_config* to *epsetup*.

    Reads optional ``sensor_position``/``sensor_direction`` from
    ``calculate_illuminance`` entries and ``view_position``/``view_direction``
    from ``calculate_edgps`` entries.  When present, overrides the
    auto-generated sensor/view with user-defined placement.

    After modifying rconfigs, affected zone workflows are recreated so that
    ThreePhaseMethod picks up the new sensor/view senders.

    Returns:
        Set of zone names whose workflows were recreated.
    """
    modified_zones: set[str] = set()

    for var_id, var_info in epsetup_config.items():
        if not isinstance(var_info, dict):
            continue

        # Custom illuminance sensor placement
        if "calculate_illuminance" in var_info:
            illum_cfg = var_info["calculate_illuminance"]
            if "sensor_position" in illum_cfg and "sensor_direction" in illum_cfg:
                zone = illum_cfg["zone"]
                pos = illum_cfg["sensor_position"]
                dir_ = illum_cfg["sensor_direction"]
                sensor_data = [pos + dir_]  # [x, y, z, dx, dy, dz]
                # Add as a NEW sensor keyed by var_id so it doesn't
                # overwrite the default WPI grid sensor.
                epsetup.rconfigs[zone].model.sensors[var_id] = SensorConfig(data=sensor_data)
                modified_zones.add(zone)
                print(f"Custom illuminance sensor '{var_id}' at {pos} dir {dir_} in zone '{zone}'")

        # Custom eDGP view/sensor placement
        if "calculate_edgps" in var_info:
            edgp_cfg = var_info["calculate_edgps"]
            if "view_position" in edgp_cfg and "view_direction" in edgp_cfg:
                zone = edgp_cfg["zone"]
                pos = edgp_cfg["view_position"]
                dir_ = edgp_cfg["view_direction"]

                first_view_key = next(iter(epsetup.rconfigs[zone].model.views.keys()))
                view = pr.create_default_view()
                view.vp = tuple(pos)
                view.vdir = tuple(dir_)
                view.type = "a"
                view.horiz = 180
                view.vert = 180
                epsetup.rconfigs[zone].model.views[first_view_key] = ViewConfig(view=view)

                sensor_data = [pos + dir_]
                epsetup.rconfigs[zone].model.sensors[first_view_key] = SensorConfig(data=sensor_data)
                modified_zones.add(zone)
                print(f"Custom eDGP view (replaced '{first_view_key}') at {pos} dir {dir_} in zone '{zone}'")

    # Recreate workflows for modified zones so the original PhaseMethod picks
    # up new senders. Preserve the workflow class chosen at
    # ``EnergyPlusSetup(radiance_method=...)`` (default 3PM, optionally 5PM).
    # Earlier versions hard-coded ``fr.ThreePhaseMethod`` here, which silently
    # downgraded any 5PM setup back to 3PM — see PR/issue tracker.
    for zone in modified_zones:
        epsetup.rconfigs[zone].settings.overwrite = True
        workflow_cls = type(epsetup.rworkflows[zone])
        epsetup.rworkflows[zone] = workflow_cls(epsetup.rconfigs[zone])

    return modified_zones


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
    def __init__(self, output_dir=None, config_file=None, run_annual=False, cleanup=True, run_period=None, treat_weather_as_actual=False, weather_files_path=None, run_periods=None, number_of_timesteps_per_hour=1, enable_radiance=True, staggered_start: bool = False, env_id: int = 0, n_envs: int = 1, weekday_from_ep: bool = False):
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
            staggered_start (bool): If True, stagger the start month deterministically based on env_id and reset count.
            env_id (int): The ID of the environment, used for staggering start months to decorrelate training data.
            n_envs (int): Total number of parallel environments. Used together with env_id to spread start months
                          evenly across the year. Defaults to 1 (no spreading).
        """
        # simulation settings
        self.run_annual = run_annual
        self.run_period = run_period
        self.treat_weather_as_actual=treat_weather_as_actual
        self.number_of_timesteps_per_hour=number_of_timesteps_per_hour
        self.action_values = None 
        self.weather_files_path=weather_files_path
        self.run_periods=run_periods  # List of dicts parallel to weather_files_path
        self.current_weather_idx = 0
        self.staggered_start = staggered_start
        self.weekday_from_ep = weekday_from_ep  # Wochentag aus EnergyPlus in die Beobachtung
        self.env_id = env_id
        self.n_envs = n_envs
        self.reset_count = 0

        self.enable_radiance = enable_radiance
        self.config_file = config_file
        self.simulation_finished = False
        self.callback_error = None  # exception raised in the EnergyPlus callback, re-raised by steps()

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
                # Relative paths in a dict config resolve against config_dir; without it they are an error
                self.input_dir = loaded_config.get('config_dir')
            else:
                 raise ValueError("config_file must be a string path or a dictionary key-value map.")
        else:
            raise ValueError("config_file is required (a path to a JSON file or a dict)")

        # Set the working directory for simulation outputs
        if output_dir is None:
            raise ValueError("output_dir is required")
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
        check_frads_capabilities(self.epsetup_config)
        self._model_setup()
        self._simulation_config()

    def _resolve(self, path):
        """Absolute path for a path from the config; relative ones resolve against the config directory."""
        if os.path.isabs(path):
            return path
        if self.input_dir is None:
            raise ValueError(f"Relative path '{path}' needs 'config_dir' in the config dict, or use absolute paths")
        return os.path.join(self.input_dir, path)

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
        self.model_file = None  # geladene Modelldatei, Schluessel fuer feste Matrizen
        
        # Load model based on available configuration
        if model_path:
            full_model_path = self._resolve(model_path)
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"model_setup.model_path not found: {full_model_path}")
            print(f"Loading model from path: {full_model_path}")
            self.epmodel = fr.load_energyplus_model(full_model_path)
            self.model_file = full_model_path
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
        # Retain the loaded GlazingSystem objects so their per-state per-layer
        # solar absorptances are available at runtime (e.g. for cfs_thermal).
        self.glazing_systems = {}
        for glazing in model_setup.get("glazing_systems", []):
            gs_path = self._resolve(glazing["file"])
            if not os.path.exists(gs_path):
                raise FileNotFoundError(f"Glazing system file not found: {gs_path}")

            # Load and add glazing system
            # gs = fr.GlazingSystem.from_json(gs_path) # frads version V1
            gs = fr.load_glazing_system(gs_path) # frads version V2
            self.epmodel.add_glazing_system(gs)
            self.glazing_systems[gs.name] = gs

            print(f"Added glazing system from {gs_path}")

        states = [s["name"] for s in self.epsetup_config.get("facade_state_mapping", {}).get("states", [])]
        unknown = [s for s in states if s not in self.glazing_systems]
        if unknown:
            raise ValueError(f"facade_state_mapping names {unknown} match no glazing system "
                             f"(loaded: {sorted(self.glazing_systems)})")

        # Configure CFS transient thermal systems (delayed absorbed-solar gain).
        # Declares one OtherEquipment object per entry; the per-window RC models
        # and the per-timestep injection are handled in the controller callback.
        self.cfs_thermal_models = {}   # window key -> CFSThermalModel (state persists across steps)
        for entry in model_setup.get("cfs_thermal_systems", []):
            add_other_equipment(
                self.epmodel,
                name=entry["equipment_name"],
                zone=entry["zone"],
                fraction_radiant=entry.get("fraction_radiant", 0.6),
                replace=entry.get("replace", True),
            )
            print(f"Added CFS thermal OtherEquipment '{entry['equipment_name']}' to zone {entry['zone']}")

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

    def _wait_for_simulation_to_finish(self, timeout=600):
        """
        Wait until the simulation signals completion.

        This method blocks until the simulation thread signals that it has
        completed execution, either normally or due to an error.

        Args:
            timeout: Maximum seconds to wait before giving up.
        """
        start = time.time()
        # Signal the simulation to shut down and wake the controller. With the
        # shutdown flag set, the controller callback's bounded put()/wait()
        # (see end of `controller`) return instead of blocking, so EnergyPlus
        # can regain control and observe the stop request below.
        self.shutdown_event.set()
        self.next_step_event.set()

        # Ask EnergyPlus to halt so _run_simulation()'s epsetup.run() returns
        # and the background thread can be joined BEFORE the caller's
        # epsetup.close() -> api.state_manager.delete_state(). Calling
        # delete_state() while that run() thread is still live blocks the main
        # thread forever in a futex (observed 2026-06-06 deadlock at 99.1%).
        if hasattr(self, 'epsetup'):
            try:
                self.epsetup.api.runtime.stop_simulation(self.epsetup.state)
            except Exception as e:
                print(f"Warning: stop_simulation request failed: {e}")

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

        # Join the EnergyPlus thread before the caller calls epsetup.close().
        # If it refuses to exit (truly stuck), leave it as an abandoned daemon
        # and flag the caller to SKIP close() so delete_state() cannot deadlock.
        self.ep_thread_alive_after_wait = False
        thread = getattr(self, 'simulation_thread', None)
        if thread is not None and thread.is_alive():
            join_deadline = time.time() + 30
            while thread.is_alive() and time.time() < join_deadline:
                self._drain_queues()
                thread.join(timeout=1.0)
            if thread.is_alive():
                self.ep_thread_alive_after_wait = True
                print("Warning: EnergyPlus thread did not exit after stop request; "
                      "abandoning it (daemon) and SKIPPING close() to avoid a "
                      "delete_state() deadlock. A fresh state will be created.")
        print("Simulation appears to be finished.")

    def _drain_queues(self):
        """Drain all pending items from communication queues."""
        for q in (self.obs_data_queue, self.action_data_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _apply_custom_radiance_config(self):
        """Apply custom sensor/view positions from epsetup_config to Radiance rconfigs.

        Delegates to the module-level apply_custom_radiance_config so that
        callers outside the environment (e.g. matrix regeneration) can reuse it.
        """
        apply_custom_radiance_config(self.epsetup, self.epsetup_config)

    def _use_fixed_matrices(self, root):
        """Feste Radiance-Matrizen (Config model_setup.fixed_matrices oder FRADS_MATRIX_DIR, seit 22.10.2026).

        frads tastet die Sensor-Fenster-Matrizen zufaellig ab; ohne festen Satz zieht jeder Lauf
        (die Zone mit eigenem Sensor sogar jeder Reset) eine eigene Realisierung, am Blendungspunkt
        etwa +-3 % E_v. Hier liest jede Zone <root>/<IDF-Name>/<Zone>.npz eines einmal erzeugten
        Satzes und prueft ihn gegen manifest.json (Gebaeudemodell, Sensoren, Dateien).
        FRADS_MATRIX_WRITE=1 erzeugt den Satz statt ihn zu lesen (rl/evaluation/make_fixed_matrices.py).
        """
        if not self.model_file:
            raise RuntimeError("FRADS_MATRIX_DIR braucht model_setup.model_path (kein Fallback-Modell)")
        set_dir = Path(root) / Path(self.model_file).stem
        write = os.environ.get("FRADS_MATRIX_WRITE") == "1"
        # vor dem Erzeugen/Laden gebildet, beim Schreiben fuer das Manifest aufbewahrt
        self.fixed_matrix_digests = {z: sensor_digest(wf) for z, wf in self.epsetup.rworkflows.items()}
        if write:
            set_dir.mkdir(parents=True, exist_ok=True)
        else:
            manifest = json.loads((set_dir / "manifest.json").read_text())
            if manifest["model_sha256"] != file_sha256(self.model_file):
                raise RuntimeError(f"{set_dir}: Gebaeudemodell geaendert, festen Satz neu erzeugen")
        for zone, wf in self.epsetup.rworkflows.items():
            wf.mfile = set_dir / f"{zone}.npz"
            wf.config.settings.overwrite = write
            if write:
                continue
            entry = manifest["zones"][zone]
            if entry["sensors_sha256"] != self.fixed_matrix_digests[zone]:
                raise RuntimeError(f"{set_dir}: Sensoren von {zone} passen nicht zum festen Satz")
            if entry["npz_sha256"] != file_sha256(wf.mfile):
                raise RuntimeError(f"{wf.mfile}: Pruefsumme stimmt nicht mit manifest.json")

    def _register_controller(self):
        """
        Register the controller with EnergyPlus without triggering AST analysis errors.

        Bypasses epsetup.set_callback() which uses fragile AST analysis
        (inspect.getsource + ast.parse) that crashes when keyword arguments
        use variables instead of string literals. Since _register_variables()
        already handles variable registration explicitly, the AST analysis
        is redundant and we register the callback directly via the runtime API.

        API-BIAS FIX (2026-04): We register with
        `callback_end_system_timestep_after_hvac_reporting` instead of the
        previous `callback_begin_system_timestep_before_predictor`. The former
        fires AT THE END of each HVAC sub-timestep, when the system-timestep
        integrated Energy variables (e.g. `Zone Air System Sensible
        Heating/Cooling Energy`, `Heating Coil NaturalGas Energy`) hold the
        finalised contribution for that sub-step.

        The controller() function accumulates those sub-step contributions in
        a per-instance HVAC accumulator and only flushes the full observation
        (plus zone-level variables and agent step) at the end of each zone
        timestep. This recovers 100% of the HVAC energy that the previous
        begin-of-system callback was missing (it captured only ~20-90% depending
        on the HVAC sub-stepping factor — see verify_api_bias_fix.py).
        """
        self.epsetup.api.runtime.callback_end_system_timestep_after_hvac_reporting(
            self.epsetup.state, simplified_controller_wrapper
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
            if not getattr(self, 'ep_thread_alive_after_wait', False):
                try:
                    self.epsetup.close()
                except Exception as e:
                    print(f"Warning: Error closing previous EnergyPlus session: {e}")
            else:
                # The previous EnergyPlus thread is wedged in an uninterruptible
                # C call (e.g. the FullExteriorWithReflections solar-init under
                # memory pressure) and could not be stopped/joined. Abandoning it
                # and building a fresh EnergyPlusSetup in THIS process is unsafe:
                # the live daemon leaks its multi-GB EP state (a live thread is
                # never GC'd, so "leaving its state to be GC'd" never happens) and
                # a second concurrent EP run in the same process deadlocks at the
                # solar init (observed 2026-06-06 v3@99% / 2026-06-08 v5@33%, 0%
                # CPU, host swap-saturated for ~9h). Under SubprocVecEnv every env
                # owns its own subprocess, so the only clean recovery is to let
                # THIS subprocess die: the OS reclaims the leaked state, and
                # SubprocVecEnv surfaces the dead worker -> the trial fails fast
                # (CheckpointCallback artifacts survive) instead of wedging the
                # whole host. os._exit bypasses interpreter cleanup that could
                # itself hang on the stuck C thread.
                print("FATAL: EnergyPlus thread unrecoverable after stop+join; "
                      "exiting this env subprocess so a fresh process can replace "
                      "it (prevents zombie-thread leak + in-process solar-init "
                      "deadlock — see frads_wrapper reset()).", flush=True)

                # Best-effort: persist the obs-normalizer EMA state before we
                # hard-exit. os._exit bypasses all interpreter cleanup
                # (atexit / finalizers), so without this the running EMA/
                # min-max tracker state accumulated this episode is lost and
                # the replacement process cold-starts the normalizer. The gym
                # env registers `self._normalizer_save_hook` (a zero-arg
                # callable wrapping save_normalizer_state); it is optional, so
                # a missing hook is a silent no-op. The save runs in a daemon
                # thread with a bounded join so a wedged pickle can never trade
                # this exit for another hang ("kein unbegrenztes Warten").
                hook = getattr(self, "_normalizer_save_hook", None)
                if callable(hook):
                    try:
                        saver = threading.Thread(target=hook, daemon=True)
                        saver.start()
                        saver.join(timeout=10)
                        if saver.is_alive():
                            print("Warning: normalizer save hook did not finish "
                                  "within 10s; exiting without it.", flush=True)
                    except Exception as e:
                        print(f"Warning: normalizer save hook failed before "
                              f"exit: {e}", flush=True)

                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)

        # Reset all synchronization primitives for the new episode
        self.shutdown_event.clear()
        self.next_step_event.clear()
        self.simulation_finished = False
        self.callback_error = None
        self._drain_queues()

        # Reset persistent controller state that must NOT survive an episode
        # boundary. A full reset() starts a fresh EnergyPlus run (new episode,
        # possibly a different weather year / start month under staggered_start),
        # so any state carried over from the previous episode corrupts the first
        # hours of the new one and breaks episode determinism:
        #   * cfs_thermal_models — RC pane temperatures. Clearing forces the
        #     controller to rebuild each model with init_temp_c=zone-air-temp
        #     of the NEW episode instead of inheriting the old episode's end
        #     temperatures (repro: ~20 K summer->winter carryover = ~400 Wh of
        #     spurious OtherEquipment injection over the first ~3 h).
        #   * _hvac_accumulator / _hvac_substep_time_accum — a forced teardown
        #     mid zone-timestep can leave partial sub-step sums and a non-zero
        #     timer; stale residue would mis-fire the zone-boundary detection
        #     in the next episode.
        self.cfs_thermal_models = {}
        self._hvac_accumulator = {}
        self._hvac_substep_time_accum = 0.0

        # Select the appropriate weather file
        if hasattr(self, 'weather_files_path') and self.weather_files_path and len(self.weather_files_path) > 0:
            current_weather = self.weather_files_path[self.current_weather_idx]
            # Rotate run_period alongside weather file if per-weather run_periods are defined
            if self.run_periods and len(self.run_periods) > self.current_weather_idx:
                base_run_period = self.run_periods[self.current_weather_idx].copy()
                
                if self.staggered_start:
                    # Proportional staggering:
                    # Spread start months evenly across all parallel environments by scaling env_id
                    # proportionally to the total number of environments. This ensures that even with
                    # only 3 envs, their start months are spread across Winter, Summer, and Autumn
                    # rather than starting in Jan, Feb and Mar (N_envs consecutive winter months).
                    # reset_count adds a +1 month shift per episode to ensure temporal diversity.
                    import math
                    base_offset = math.floor(self.env_id * 12 / self.n_envs)
                    offset_months = (base_offset + self.reset_count) % 12
                    
                    # Original begin month
                    orig_begin_month = int(base_run_period.get("begin_month", 1))
                    
                    # Compute new begin month (1-indexed)
                    new_begin_month = ((orig_begin_month - 1 + offset_months) % 12) + 1
                    
                    # Since we are assuming _2year EPW files for staggered training, 
                    # we always run for exactly 12 months.
                    new_end_month = ((new_begin_month - 2) % 12) + 1  # end month is month before begin month
                    
                    # Calculate year boundaries
                    orig_begin_year = int(base_run_period.get("begin_year", 2002))
                    
                    # Start year is the base year
                    new_begin_year = orig_begin_year
                    # End year depends on whether we crossed the year boundary 
                    # (if new_end_month < new_begin_month, we crossed into the next year)
                    if new_end_month < new_begin_month:
                        new_end_year = new_begin_year + 1
                    else:
                        new_end_year = new_begin_year
                    
                    # Update periods
                    base_run_period["begin_month"] = new_begin_month
                    base_run_period["end_month"] = new_end_month
                    base_run_period["begin_year"] = new_begin_year
                    base_run_period["end_year"] = new_end_year
                    # To keep it simple, we start on day 1 and end on the common last day of the month by defaulting to 28 for testing
                    # EnergyPlus handles end_day nicely, but we use typical month boundaries
                    base_run_period["begin_day"] = 1
                    
                    import calendar
                    # Find out the last day of the new_end_month
                    last_day_end_month = calendar.monthrange(new_end_year, new_end_month)[1]
                    base_run_period["end_day"] = last_day_end_month
                    
                    self.reset_count += 1
                    print(f"Staggered start enabled (env_id={self.env_id}, reset={self.reset_count-1}): Offset {offset_months} months -> starting {new_begin_month}/{new_begin_year} to {new_end_month}/{new_end_year}")

                self.run_period = base_run_period
                print(f"Using run_period: {self.run_period}")
            self._active_weather_idx = self.current_weather_idx
            self.current_weather_idx = (self.current_weather_idx + 1) % len(self.weather_files_path)
            print(f"Using weather file: {current_weather} (index {self._active_weather_idx})")
        else:
            raise ValueError("weather_files_path is required: a non-empty list of EPW files")

        # Update EnergyPlus model with current run_period before starting simulation
        self._simulation_config()

        # Initialize EnergyPlus Simulation Setup with Radiance enabled
        # Use initialize_radiance=False to allow custom sensor/view placement before matrix generation
        self.epsetup = fr.EnergyPlusSetup(
            self.epmodel,
            current_weather,
            enable_radiance=self.enable_radiance,
            initialize_radiance=False
        )

        # Apply custom sensor/view positions from config, then initialize Radiance
        if self.enable_radiance:
            self._apply_custom_radiance_config()
            fixed = os.environ.get("FRADS_MATRIX_DIR") or self.epsetup_config.get("model_setup", {}).get("fixed_matrices")
            if fixed:
                self._use_fixed_matrices(self._resolve(fixed))
            # The 5PM-image glare path (FivePhaseMethod.calculate_dgp)
            # requires the view matrices (V, Vd, Cds, Cdf) which
            # initialize_radiance skips by default. Promote when any
            # config entry routes to calculate_dgp -- otherwise we'd
            # bury the user in NoneType subscript errors at runtime.
            needs_view_matrices = any(
                isinstance(v, dict) and "calculate_dgp" in v
                for v in self.epsetup_config.values()
            )
            if needs_view_matrices:
                self.epsetup.initialize_radiance(view_matrices=True)
            else:
                self.epsetup.initialize_radiance()

        # Set the simulation instance in the global context singleton
        context.simulation = self

        # Register the controller callback for each timestep
        self._register_controller()

        # Start the simulation in a background thread. Stored on self so
        # _wait_for_simulation_to_finish() can join it before close().
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()

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
            # Signal that the simulation has ended. Bounded put: if the main
            # thread already stopped draining (forced teardown), an unguarded
            # put on the maxsize=1 queue would block this thread forever and
            # prevent it from being joined.
            try:
                self.obs_data_queue.put({'simulation_finished': True}, timeout=10)
            except queue.Full:
                pass
            
    def steps(self, action_data=None, release=True):
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
        self._raise_callback_error()
        if self.simulation_finished:
            return {'simulation_finished': True}

        try:
            # Send action data to controller. Bounded timeout mirrors the
            # obs_data_queue.get() below: when EnergyPlus finishes, its
            # controller callback thread stops draining action_data_queue
            # (maxsize=1), so an unguarded put() on a full queue would block
            # forever at teardown. Treat a full queue as end-of-simulation.
            if action_data:
                self.action_data_queue.put(action_data, timeout=600)

            # Signal controller to continue. release=False holt nur die naechste Beobachtung, ohne den
            # Controller freizugeben (erste Beobachtung nach dem EnergyPlus-Start, sync_first_action).
            if release:
                self.next_step_event.set()

            # Wait for data from controller. 600s (was 120s, then 300s): under
            # multi-env contention the one-time EnergyPlus solar/shadowing init
            # (FullExteriorWithReflections over ~1465 shading surfaces, started
            # by 4 EP processes at once) and the periodic 30-day shadow recompute
            # can legitimately stall >300s. The longer bound lets such steps
            # complete instead of returning a key-less {'error'} obs.
            self.obs_data = self.obs_data_queue.get(timeout=600)
            self._raise_callback_error()

            # Check if the simulation has ended
            if isinstance(self.obs_data, dict) and self.obs_data.get('simulation_finished', False):
                self.simulation_finished = True
                return {'simulation_finished': True}

            return self.obs_data
        except queue.Full:
            print("Warning: action queue full for 600s — EnergyPlus consumer "
                  "gone, treating as simulation finished.")
            self.simulation_finished = True
            return {'simulation_finished': True}
        except queue.Empty:
            self._raise_callback_error()
            print("Warning: No observation data received within timeout (600s).")
            return {'error': 'No step data available (timeout)'}

    def _raise_callback_error(self):
        if self.callback_error is not None:
            self.simulation_finished = True
            raise RuntimeError(f"Error in the EnergyPlus callback: {self.callback_error}") from self.callback_error

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

        # Properly close EnergyPlus simulation (unless the thread is still alive,
        # in which case close() -> delete_state() would deadlock — see
        # _wait_for_simulation_to_finish).
        if hasattr(self, 'epsetup') and not getattr(self, 'ep_thread_alive_after_wait', False):
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

        # Register variables needed by cfs_thermal entries (the controller reads
        # these with get_variable_value, which uses non-literal keys, so the
        # AST-based auto-request cannot pick them up — request them explicitly).
        for var_id, var_info in self.epsetup_config.items():
            if "cfs_thermal" in var_info:
                ct = var_info["cfs_thermal"]
                for win in ct.get("windows", {}):
                    self.epsetup.request_variable(
                        "Surface Outside Face Incident Solar Radiation Rate per Area", win)
                    self.epsetup.request_variable(
                        "Surface Inside Face Convection Heat Transfer Coefficient", win)
                    if ct.get("absorptance_mode") == "angular":
                        self.epsetup.request_variable(
                            "Surface Outside Face Beam Solar Incident Angle Cosine Value", win)
                self.epsetup.request_variable("Zone Mean Air Temperature", ct["zone"])
                self.epsetup.request_variable("Site Outdoor Air Drybulb Temperature", "Environment")
                print(f"Registered cfs_thermal variables for {var_id} ({len(ct.get('windows', {}))} windows)")


# ---------------------------------------------------------------------------
# API-BIAS FIX — HVAC accumulator state (per-simulation, module-level helpers)
# ---------------------------------------------------------------------------
#
# Variables whose name matches one of these patterns are SYSTEM-timestep
# integrated (one chunk per HVAC sub-step). They must be accumulated across
# sub-steps inside a zone timestep and flushed at the zone-timestep boundary.
#
# Everything else (Zone Electric Equipment, Zone People, Zone Lights,
# Surface Window Transmitted Solar Radiation, etc.) is ZONE-timestep
# integrated — read it fresh once per zone timestep.
#
# The accumulator and sub-step timer are attached to the simulation instance
# in controller() on first use.
# ---------------------------------------------------------------------------

_HVAC_SUBSTEP_VAR_PATTERNS = (
    "Zone Air System Sensible Heating Energy",
    "Zone Air System Sensible Cooling Energy",
    "Zone Air System Latent Heating Energy",
    "Zone Air System Latent Cooling Energy",
    "Zone Air System Total Heating Energy",
    "Zone Air System Total Cooling Energy",
    "Zone Mechanical Ventilation Heating Load Increase Energy",
    "Zone Mechanical Ventilation Cooling Load Increase Energy",
    "Zone Mechanical Ventilation No Load Heat Addition Energy",
    "Zone Mechanical Ventilation No Load Heat Removal Energy",
    "Heating Coil NaturalGas Energy",
    "Heating Coil Electricity Energy",
    "Heating Coil Air Heating Energy",
    "Heating Coil Total Heating Energy",
    "Cooling Coil Electricity Energy",
    "Cooling Coil Total Cooling Energy",
    "Cooling Coil Sensible Cooling Energy",
    "Cooling Coil Latent Cooling Energy",
    "Fan Electricity Energy",
    "Pump Electricity Energy",
)


def _is_hvac_substep_variable(var_name: str) -> bool:
    """Return True if this EP variable is integrated per HVAC system sub-timestep.

    Such variables must be accumulated across sub-steps inside a zone
    timestep to recover the full zone-timestep total. Override per-variable
    via the config field `"hvac_substep_accumulate": true/false`.
    """
    return any(p in var_name for p in _HVAC_SUBSTEP_VAR_PATTERNS)


def controller(state):
    """
    Controller function called at each EnergyPlus timestep.
    Uses the epsetup_config to dynamically collect observations.

    API-BIAS FIX (2026-04): registered with
    `end_system_timestep_after_hvac_reporting`, meaning this function fires
    ONCE per HVAC system sub-timestep. The sub-step count per zone timestep
    can be 1 to ~10 depending on HVAC solver convergence. For HVAC-energy
    variables we accumulate across sub-steps; for everything else we only
    read/flush at the zone-timestep boundary.
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

    # Lazy-init HVAC accumulator state on the simulation instance
    if not hasattr(simulation, "_hvac_accumulator"):
        simulation._hvac_accumulator = {}
        simulation._hvac_substep_time_accum = 0.0  # hours accumulated in current zone TS

    # ------------------------------------------------------------------
    # Phase 1: accumulate HVAC sub-step contributions (runs every sub-step)
    # ------------------------------------------------------------------
    exchange = simulation.epsetup.api.exchange
    sys_ts_hours = exchange.system_time_step(state)   # fractional hour
    zone_ts_hours = exchange.zone_time_step(state)    # fractional hour

    for var_id, var_info in simulation.epsetup_config.items():
        gv = var_info.get("get_variable_value") if isinstance(var_info, dict) else None
        if not gv:
            continue
        var_name = gv["name"]
        var_key = gv["key"]
        # Per-variable override via config (optional; defaults to name-based detection)
        override = var_info.get("hvac_substep_accumulate")
        is_hvac = override if isinstance(override, bool) else _is_hvac_substep_variable(var_name)
        if is_hvac:
            # The handle may be missing for variables this IDF/HVAC configuration does not
            # instantiate (e.g. latent cooling energy without humidity control).
            try:
                val = simulation.epsetup.get_variable_value(name=var_name, key=var_key)
            except Exception as exc:
                raise RuntimeError(f"'{var_id}': EnergyPlus variable {var_name!r} for key {var_key!r} "
                                   f"is not available in this model") from exc
            simulation._hvac_accumulator[var_id] = (
                simulation._hvac_accumulator.get(var_id, 0.0) + float(val)
            )

    simulation._hvac_substep_time_accum += float(sys_ts_hours)

    # ------------------------------------------------------------------
    # Phase 2: zone-timestep-boundary check
    # Only flush the full observation pipeline once per zone timestep
    # ------------------------------------------------------------------
    # Floating-point tolerance: accept if we've accumulated within 1e-6 hours
    # (= 3.6 ms) of a full zone timestep
    at_zone_boundary = (
        simulation._hvac_substep_time_accum >= float(zone_ts_hours) - 1e-6
    )
    if not at_zone_boundary:
        # Not yet at the zone timestep end; wait for more sub-steps
        return

    # Reset the sub-step timer for the next zone timestep
    simulation._hvac_substep_time_accum = 0.0

    ###################################
    # Initialize observation data with datetime
    obs_data = {'datetime': simulation.epsetup.get_datetime()}
    # Wochentag nach dem Kalender der RunPeriod (1=Sonntag ... 7=Samstag), nach dem
    # EnergyPlus die Belegung plant; das Datum oben traegt das Quelljahr der Wetterdatei.
    if simulation.weekday_from_ep:
        obs_data['day_of_week'] = simulation.epsetup.api.exchange.day_of_week(simulation.epsetup.state)

    # 1. First process all get_variable_value entries
    #    HVAC-substep variables use the accumulator (full zone-TS total);
    #    zone-level variables are read fresh now.
    for var_id, var_info in simulation.epsetup_config.items():
        if "get_variable_value" in var_info:
            var_name = var_info["get_variable_value"]["name"]
            var_key = var_info["get_variable_value"]["key"]
            override = var_info.get("hvac_substep_accumulate")
            is_hvac = override if isinstance(override, bool) else _is_hvac_substep_variable(var_name)
            if is_hvac:
                obs_data[var_id] = simulation._hvac_accumulator.get(var_id, 0.0)
            else:
                try:
                    obs_data[var_id] = simulation.epsetup.get_variable_value(
                        name=var_name,
                        key=var_key,
                    )
                except Exception as exc:
                    raise RuntimeError(f"'{var_id}': EnergyPlus variable {var_name!r} for key {var_key!r} "
                                       f"is not available in this model") from exc

    # Reset HVAC accumulator for the next zone timestep
    simulation._hvac_accumulator = {}
    
    # 1.5 Process all calculate_allocation entries (proportional energy split)
    for var_id, var_info in simulation.epsetup_config.items():
        if "calculate_allocation" in var_info:
            alloc_cfg = var_info["calculate_allocation"]
            missing = [alloc_cfg[k] for k in ("source_key", "numerator_key", "denominator_key")
                       if alloc_cfg[k] not in obs_data]
            if missing:
                raise RuntimeError(f"'{var_id}': calculate_allocation needs {missing}, which are computed "
                                   "later or not at all (use get_variable_value entries)")
            source = float(obs_data[alloc_cfg["source_key"]])
            numerator = float(obs_data[alloc_cfg["numerator_key"]])
            denominator = float(obs_data[alloc_cfg["denominator_key"]])
            if denominator > 0:
                obs_data[var_id] = source * (numerator / denominator)
            else:
                obs_data[var_id] = 0.0

    # 2. Process all get_cfs_state entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "get_cfs_state" in var_info:
            window_key = var_info["get_cfs_state"]["key"]
            obs_data[var_id] = simulation.epsetup.get_cfs_state(window_key)

    # 2.5 Process all cfs_thermal entries: advance a transient lumped-capacitance
    #     RC glass model per window (state persists), and inject the ENERGY-NEUTRAL
    #     delayed absorbed-solar inward gain (transient - steady) as an
    #     OtherEquipment power level. Transmission stays in the optical CFS path.
    for var_id, var_info in simulation.epsetup_config.items():
        if "cfs_thermal" not in var_info:
            continue
        ct = var_info["cfs_thermal"]
        dt_s = float(zone_ts_hours) * 3600.0
        try:
            t_in = simulation.epsetup.get_variable_value("Zone Mean Air Temperature", ct["zone"])
            t_out = simulation.epsetup.get_variable_value(
                "Site Outdoor Air Drybulb Temperature", "Environment")
        except Exception as exc:
            raise RuntimeError(f"'{var_id}': cfs_thermal could not read the zone or outdoor temperature") from exc
        total_gain_w = 0.0
        pane_temps = None
        for win, area in ct["windows"].items():
            try:
                cfs_name = simulation.epsetup.get_cfs_state(win)
                gs_name = ct.get("state_to_gs", {}).get(cfs_name, cfs_name)
                gs = simulation.glazing_systems.get(gs_name)
                if gs is None:
                    raise KeyError(f"no glazing system '{gs_name}' for state '{cfs_name}' "
                                   f"(loaded: {sorted(simulation.glazing_systems)}; see 'state_to_gs')")
                model = simulation.cfs_thermal_models.get(win)
                if model is None:
                    model = CFSThermalModel.from_glazing_system(
                        gs,
                        overrides={int(k): v for k, v in ct.get("pane_overrides", {}).items()},
                        h_out=ct.get("h_out", 23.0),
                        h_in=ct.get("h_in", 8.0),
                        gap_nu=ct.get("gap_nu", 1.0),
                        absorptance_mode=ct.get("absorptance_mode", "hemispherical"),
                        init_temp_c=float(t_in),
                    )
                    model._gs_name = gs_name
                    simulation.cfs_thermal_models[win] = model
                elif getattr(model, "_gs_name", None) != gs_name:
                    model.set_absorptance(gs.solar_front_absorptance)  # tint switched; keep T
                    model._gs_name = gs_name
                incident = simulation.epsetup.get_variable_value(
                    "Surface Outside Face Incident Solar Radiation Rate per Area", win)
                try:
                    h_in = simulation.epsetup.get_variable_value(
                        "Surface Inside Face Convection Heat Transfer Coefficient", win)
                except Exception as exc:
                    raise RuntimeError("Surface Inside Face Convection Heat Transfer Coefficient "
                                       f"is not available for {win}") from exc
                res = model.step(float(incident), float(t_out), float(t_in), dt_s, h_in=float(h_in))
                total_gain_w += res.lag_correction_wm2 * float(area)
                pane_temps = res.pane_temps_c
            except Exception as exc:
                raise RuntimeError(f"'{var_id}': cfs_thermal failed for window {win}") from exc
        obs_data[var_id] = total_gain_w
        if pane_temps is not None and var_info.get("log_pane_temps"):
            for k, t in enumerate(pane_temps):
                obs_data[f"{var_id}_pane{k}"] = t
        # Inject now (sets the OtherEquipment power for the next zone timestep,
        # same latency as actuate_cfs_state). Negative values are allowed.
        try:
            simulation.epsetup.actuate(
                "OtherEquipment", "Power Level", ct["equipment_name"], total_gain_w)
        except Exception as exc:
            raise RuntimeError(f"'{var_id}': OtherEquipment 'Power Level' actuator for "
                               f"{ct['equipment_name']!r} is not available") from exc

    # 3. Process all calculate_illuminance entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "calculate_illuminance" in var_info:
            illum_cfg = var_info["calculate_illuminance"]
            zone = illum_cfg["zone"]
            cfs_names = illum_cfg["cfs_name"]

            # Create cfs_name dictionary mapping window names to their current states
            cfs_dict = {}
            for window in cfs_names:
                cfs_dict[window] = simulation.epsetup.get_cfs_state(window)

            # Use custom sensor if sensor_position/sensor_direction were configured,
            # otherwise fall back to default WPI grid via calculate_wpi.
            #
            # Errors raised in this callback reach the main thread through
            # simplified_controller_wrapper and FradsSimulation.steps().
            try:
                if "sensor_position" in illum_cfg and "sensor_direction" in illum_cfg:
                    # Custom sensor was registered under var_id by apply_custom_radiance_config
                    date_time = simulation.epsetup.get_datetime()
                    dni = simulation.epsetup.get_direct_normal_irradiance()
                    dhi = simulation.epsetup.get_diffuse_horizontal_irradiance()
                    wf = simulation.epsetup.rworkflows[zone]
                    if os.environ.get("FRADS_SENSOR_DEBUG") == "1":
                        print(f"[CALLER_CALC_SENSOR] var_id={var_id} workflow_type={type(wf).__name__} method_qualname={wf.calculate_sensor.__qualname__}", flush=True)
                    wpi_result = wf.calculate_sensor(
                        var_id, cfs_dict, date_time, dni, dhi
                    )
                else:
                    wpi_result = simulation.epsetup.calculate_wpi(zone=zone, cfs_name=cfs_dict)

                # Apply post-processing if specified
                if "post_processing" in var_info and var_info["post_processing"] == "mean":
                    obs_data[var_id] = wpi_result.mean()
                else:
                    obs_data[var_id] = wpi_result
            except Exception as exc:
                raise RuntimeError(f"'{var_id}': illuminance calculation failed in zone {zone}") from exc

    # 3.5. Apply per-sensor origin-constrained calibration to illuminance.
    # real_estimate = k * sim_raw  (intercept fixed at 0). k is per-sensor
    # and loaded from epsetup_config[var_id]["calibration"]; missing/k=1.0
    # is a no-op. Applied BEFORE Phase 4 so calculate_dgp's ev_sensor lookup
    # (which re-runs calculate_sensor internally) can be paired with an
    # ev_override at the call site. Reward functions that consume Ev
    # post-hoc (e.g. comfort_glare_dgps Wienold formula on wpi_cam_1)
    # automatically pick up the calibrated value via info[...] / obs.
    import numpy as _np_cal
    for var_id, var_info in simulation.epsetup_config.items():
        cal = var_info.get("calibration") if isinstance(var_info, dict) else None
        if not isinstance(cal, dict):
            continue
        if cal.get("method") != "origin_scaling":
            continue
        if var_id not in obs_data:
            continue
        try:
            k = float(cal.get("k", 1.0))
        except (TypeError, ValueError):
            continue
        if k == 1.0:
            continue
        raw = obs_data[var_id]
        try:
            if isinstance(raw, tuple):
                # eDGP tuple (dgp, ev) — caller will overwrite later in
                # Phase 4 with calculate_dgp result, so leave this branch
                # for safety only.
                continue
            if isinstance(raw, _np_cal.ndarray):
                obs_data[var_id] = _np_cal.maximum(0.0, k * raw.astype(float))
            elif hasattr(raw, "__len__"):
                obs_data[var_id] = [max(0.0, k * float(x)) for x in raw]
            else:
                obs_data[var_id] = max(0.0, k * float(raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"'{var_id}': calibration k={k} cannot be applied to {raw!r}") from exc

    # 4. Process all calculate_edgps / calculate_dgp entries
    #
    # Two glare metrics are supported:
    #   - calculate_dgp:  5PM matrix image + evalglare (McNeil 2013).
    #                     Scientifically rigorous for BSDF facades --
    #                     matches the workflow's calculate_sensor (WPI/Ev)
    #                     numerical model.
    #   - calculate_edgps: rpict fisheye + evalglare (Wienold 2009 eDGPs).
    #                      Older simplified path; rpict cannot resolve
    #                      Klems-BSDF transmission, so DGP saturates for
    #                      BSDF facades. Kept for backward compatibility.
    #
    # If a var_info dict has BOTH keys (config in transition) the 5PM
    # path wins -- there's no scenario where running both makes sense
    # for the same observation slot.
    #
    # Errors raised here reach the main thread through simplified_controller_wrapper.
    for var_id, var_info in simulation.epsetup_config.items():
        cfg = var_info.get("calculate_dgp") or var_info.get("calculate_edgps")
        if cfg is None:
            continue
        zone = cfg["zone"]
        cfs_names = cfg["cfs_name"]

        # Create cfs_name dictionary mapping window names to their current states
        cfs_dict = {}
        for window in cfs_names:
            cfs_dict[window] = simulation.epsetup.get_cfs_state(window)

        _sensor_debug = os.environ.get("FRADS_SENSOR_DEBUG") == "1"
        try:
            if "calculate_dgp" in var_info:
                if _sensor_debug:
                    print(f"[DGP_ROUTE] var_id={var_id} path=calculate_dgp ev_sensor={cfg.get('ev_sensor')}", flush=True)
                obs_data[var_id] = simulation.epsetup.calculate_dgp(
                    zone=zone,
                    cfs_name=cfs_dict,
                    ev_sensor=cfg.get("ev_sensor"),
                    save_hdr=cfg.get("save_hdr"),
                )
                if _sensor_debug:
                    print(f"[DGP_RESULT] var_id={var_id} val={obs_data[var_id]!r}", flush=True)
            else:
                if _sensor_debug:
                    print(f"[DGP_ROUTE] var_id={var_id} path=calculate_edgps (NO ev_sensor passed)", flush=True)
                obs_data[var_id] = simulation.epsetup.calculate_edgps(
                    zone=zone,
                    cfs_name=cfs_dict,
                )
                if _sensor_debug:
                    print(f"[EDGP_RESULT] var_id={var_id} val={obs_data[var_id]!r}", flush=True)
        except Exception as exc:
            label = "DGP" if "calculate_dgp" in var_info else "eDGP"
            raise RuntimeError(f"'{var_id}': {label} calculation failed in zone {zone}") from exc

        # 4.5. Calibrate Ev component of DGP/eDGP tuple using the configured
        # ev_sensor's calibration.k. This keeps the Wienold-DGPs reward path
        # (info['raw_next_edgps_1'][1] as Ev) consistent with the per-sensor
        # cal scale applied in Phase 3.5. The DGP component (raw[0]) is left
        # unchanged: it was computed by evalglare from uncalibrated HDR
        # pixels, so scaling it post-hoc is not physically meaningful; for
        # the cam Ev sensor k_ev=0.996 the DGP error stays under ~1 %.
        ev_sensor_name = cfg.get("ev_sensor")
        if ev_sensor_name:
            ev_cfg = simulation.epsetup_config.get(ev_sensor_name)
            ev_cal = ev_cfg.get("calibration") if isinstance(ev_cfg, dict) else None
            if isinstance(ev_cal, dict) and ev_cal.get("method") == "origin_scaling":
                try:
                    k_ev = float(ev_cal.get("k", 1.0))
                except (TypeError, ValueError):
                    k_ev = 1.0
                if k_ev != 1.0:
                    raw_pair = obs_data.get(var_id)
                    if isinstance(raw_pair, tuple) and len(raw_pair) == 2:
                        try:
                            dgp_v = float(raw_pair[0])
                            ev_v = float(raw_pair[1])
                            obs_data[var_id] = (dgp_v, max(0.0, k_ev * ev_v))
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(f"'{var_id}': calibration of Ev with k={k_ev} cannot be "
                                               f"applied to {raw_pair!r}") from exc

    # 5. Process lighting power calculations for entries that depend on other values
    for var_id, var_info in simulation.epsetup_config.items():
        if "lighting_power_calculation" in var_info and "depends_on" in var_info:
            depends_on_key = var_info["depends_on"]
            
            # Check if the dependency is available in obs_data
            if depends_on_key in obs_data:
                # Get zone from the dependent key's configuration
                zone = simulation.epsetup_config[depends_on_key]["calculate_illuminance"]["zone"]
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
                # Radiance calculate_illuminance may return nested array [[val]];
                # unwrap to scalar float for arithmetic and EP actuator
                while hasattr(wpi_value, '__len__') and len(wpi_value) > 0:
                    wpi_value = wpi_value[0]
                wpi_value = float(wpi_value)
                lighting_power = (1 - min(wpi_value / lux_threshold, 1)) * power_density

                # Apply occupancy gate if specified: lights off when zone is unoccupied
                occupancy_key = lighting_calc.get("occupancy_key")
                if occupancy_key and occupancy_key in obs_data:
                    raw_occ = obs_data[occupancy_key]
                    occupant_count = float(raw_occ[0]) if hasattr(raw_occ, '__getitem__') else float(raw_occ)
                    if occupant_count <= 0:
                        lighting_power = 0.0

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
                raise RuntimeError(f"'{var_id}': divider {divider_value!r} cannot be applied") from e
    
    # 7. Process all calculate_energy_sum entries
    for var_id, var_info in simulation.epsetup_config.items():
        if "calculate_energy_sum" in var_info:
            # Reads the updated keys_to_sum from the JSON config
            keys_to_sum = var_info["calculate_energy_sum"].get("keys_to_sum", [])
            total_energy = 0
            for key in keys_to_sum:
                value = obs_data.get(key)
                if value is None:
                    raise RuntimeError(f"'{var_id}': '{key}' is not in the observation data for the energy sum")
                try:
                    total_energy += float(value)
                except (ValueError, TypeError) as e:
                    raise RuntimeError(f"'{var_id}': '{key}' = {value!r} is not numeric") from e
            obs_data[var_id] = total_energy

    ###################################
    ###################################
    # Put data into the queue for external controller.
    # Bounded + shutdown-aware: a plain put()/wait() on the maxsize=1 queue
    # blocks this EnergyPlus thread forever if the main thread stopped draining
    # (forced teardown after a >timeout step). Re-checking shutdown_event lets
    # the thread return so EnergyPlus can honor stop_simulation() and exit,
    # instead of wedging the subsequent delete_state() (2026-06-06 deadlock).
    while not simulation.shutdown_event.is_set():
        try:
            simulation.obs_data_queue.put(obs_data, timeout=1.0)
            break
        except queue.Full:
            continue
    if simulation.shutdown_event.is_set():
        return

    # Wait for signal to continue (bounded + shutdown-aware)
    while not simulation.next_step_event.wait(timeout=1.0):
        if simulation.shutdown_event.is_set():
            return
    simulation.next_step_event.clear()
    if simulation.shutdown_event.is_set():
        return

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
    """EnergyPlus callback. ctypes swallows exceptions raised in a callback, so an error is stored,
    EnergyPlus is asked to stop and FradsSimulation.steps() re-raises it in the main thread."""
    simulation = context.simulation
    if simulation.callback_error is not None:
        return
    try:
        return controller(state)
    except Exception as exc:
        simulation.callback_error = exc
        simulation.epsetup.api.runtime.stop_simulation(state)
        try:
            simulation.obs_data_queue.put_nowait({"callback_error": exc})
        except queue.Full:
            pass  # steps() also checks callback_error after its timeout


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

