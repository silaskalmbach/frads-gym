# Configuration

A configuration is a JSON object (or a Python dict) with two fixed blocks, `model_setup` and
`facade_state_mapping`, and any number of named **entries**. Each entry produces one value per timestep
and may be observed, actuated, or only read and logged. Keys starting with `_` are ignored everywhere and
can hold comments.

When `config_file` is a path, relative paths in it resolve against the file's directory (frads-gym sets
`config_dir` and `config_file_path` itself). For a dict, set `config_dir` to that directory or use
absolute paths.

The whole configuration is checked when the environment is created; unknown keys, missing references and
inconsistent settings raise `ConfigError` with every problem listed.

## `model_setup`

| Key | Type | Meaning |
|---|---|---|
| `model_path` | path | EnergyPlus IDF or epJSON model. Required unless `model_type` is given. |
| `model_type` | string | Name of a reference model shipped with pyenergyplus, used when `model_path` is empty: `full_service_restaurant`, `hospital`, `large_hotel`, `large_office`, `medium_office`, `midrise_apartment`, `outpatient`, `primary_school`, `quick_service_restaurant`, `secondary_school`, `small_hotel`, `small_office`, `standalone_retail`, `strip_mall`, `supermarket`, `warehouse` (DOE reference buildings, new construction 2004, Chicago). |
| `glazing_systems` | list of `{"file": path}` | frads glazing-system JSON files (for example from `frads.create_glazing_system(...).save()`). Each becomes a complex fenestration state named after the glazing system. |
| `lighting_systems` | list of `{"zone", "level", "replace"}` | Adds a lighting object of `level` W to `zone` (frads `add_lighting`; `replace` defaults to true). Needed for zones whose lights `lighting_power_calculation` dims. |
| `cfs_thermal_systems` | list of `{"equipment_name", "zone", "fraction_radiant", "replace"}` | Adds an OtherEquipment object that a [`cfs_thermal`](#cfs_thermal) entry drives (`fraction_radiant` defaults to 0.6, `replace` to true). |
| `fixed_matrices` | path | Directory of a fixed Radiance matrix set ([Reproducibility](how-it-works.md#reproducibility)). Needs `model_path`. |

## `facade_state_mapping`

```json
"facade_state_mapping": {"states": [
  {"name": "ec60", "max_threshold": 0.25},
  {"name": "ec18", "max_threshold": 0.5},
  {"name": "ec06", "max_threshold": 0.75},
  {"name": "ec01", "max_threshold": 1.0}
]}
```

An action value `a` (clipped to [0, 1]) selects the first state with `a <= max_threshold`. Thresholds
must increase and the last one must be at least 1. Every `name` must be the name of a loaded glazing
system. The same mapping, inverted to the midpoint of each interval, turns an observed glazing state back
into a number. Extra keys such as `description` are allowed.

## Entries

Every entry needs exactly one **data source** (the keys in the next section), or only
`actuate_cfs_state`. The other keys are optional:

| Key | Meaning |
|---|---|
| `active` | `false` drops the entry entirely (it is not read, observed or checked). Default `true`. |
| `observation_space` | `{"low", "high"}`: the entry becomes a key of the observation space with these bounds (lists for array values). `{"active": false}` reads and logs the value without observing it. |
| `observation_normalize` | Scales the raw value into the observation bounds ([Normalisation](#normalisation)). |
| `observation_standardize` | z-score standardisation instead of `observation_normalize` ([Normalisation](#normalisation)). |
| `actuate_cfs_state` | `{"key": <window>}`: this entry adds one dimension to the action vector, which sets the glazing state of the window. Requires `action_space`. |
| `action_space` | `{"low", "high"}`: bounds of this action dimension. Action dimensions follow the order of the entries in the configuration. |
| `post_processing` | `"mean"`: average the array returned by `calculate_illuminance`. |
| `divider` | Divide the value by this number (applied before `calculate_energy_sum`). |
| `calibration` | `{"method": "origin_scaling", "k": <number>}`: multiply an illuminance by `k` and clip at 0. For a `calculate_dgp` entry, the `k` of its `ev_sensor` also scales the vertical illuminance in the result. The factor is exposed as `info["calibration_<entry>_k"]`. |
| `hvac_substep_accumulate` | `true`/`false`: sum a `get_variable_value` over the HVAC sub-steps of a zone timestep, overriding the automatic choice (on for zone air system heating/cooling energies, mechanical ventilation load energies and coil energies). |
| `depends_on` | The `calculate_illuminance` entry that `lighting_power_calculation` reads. |
| `log_pane_temps` | `true`: a `cfs_thermal` entry also reports `<entry>_pane<i>` pane temperatures. |

### Data sources

#### `get_variable_value`

`{"name": <output variable>, "key": <key>}`: an EnergyPlus output variable, for example
`{"name": "Zone Mean Air Temperature", "key": "Perimeter_mid_ZN_1"}` or
`{"name": "Site Outdoor Air Drybulb Temperature", "key": "Environment"}`. A variable the model does not
provide is an error at the first step.

#### `get_cfs_state`

`{"key": <window>}`: the current glazing state of a window. Observed entries must have `cfs_state` in
their name; the state is mapped back to a number through `facade_state_mapping` and scaled to the
observation bounds. Usually combined with `actuate_cfs_state` on the same window.

#### `calculate_illuminance`

`{"zone", "cfs_name": [<windows>]}`: illuminance in `zone` [lx] from the three-phase method, for the
current states of the listed windows. Without further keys it is computed on frads' work-plane sensor
grid and returns one value per grid point (use `"post_processing": "mean"` for the average). With
`"sensor_position": [x, y, z]` and `"sensor_direction": [dx, dy, dz]` it is computed at that point
instead, for example `[0, 0, 1]` for horizontal or `[0, -1, 0]` for vertical, south-facing illuminance.

#### `calculate_edgps`

`{"zone", "cfs_name": [<windows>]}`: enhanced simplified daylight glare probability for the zone's view,
returned as `[eDGPs, Ev]` (Ev in lx), for frads' default view of the zone or, with
`"view_position": [x, y, z]` and `"view_direction": [dx, dy, dz]`, for a 180° angular fisheye view from
that point.

#### `calculate_dgp`

Image-based DGP from the five-phase method: `{"zone", "cfs_name", "view_position", "view_direction",
"ev_sensor"}` and optionally `"save_hdr"`, returned as `[DGP, Ev]`. `ev_sensor` names a
`calculate_illuminance` entry at the eye. Needs the [frads fork](five-phase.md). An entry may carry both
`calculate_dgp` and `calculate_edgps`; `calculate_dgp` is then used.

#### `lighting_power_calculation`

Daylight-dependent dimming of the zone lights:
`{"lux_threshold", "power_density"}` (installed power in W) or
`{"lux_threshold", "power_density_m2", "area_m2"}` (W/m² and m²), with `depends_on` naming the
illuminance entry. The power is `(1 - min(E / lux_threshold, 1)) * P`, where `E` is the illuminance;
with `"occupancy_key": <entry>` it is 0 whenever that entry is 0. The lights of the zone are actuated
with this power and the entry reports the energy of the timestep [J].

#### `calculate_allocation`

`{"source_key", "numerator_key", "denominator_key"}`: `source * numerator / denominator` (0 if the
denominator is 0), for example a central coil energy split by the zone's share of the airflow. All
three must be entries computed before (`get_variable_value`).

#### `calculate_energy_sum`

`{"keys_to_sum": [<entries>]}`: the sum of these entries after their `divider`.

#### `cfs_thermal`

A transient lumped-capacitance model of the glass panes that delays the absorbed solar gain of strongly
absorbing glazing, which EnergyPlus treats as massless. It injects the difference between the transient
and the steady inward gain through the OtherEquipment object of `model_setup.cfs_thermal_systems`:

| Key | Meaning |
|---|---|
| `equipment_name`, `zone` | The OtherEquipment object and its zone. |
| `windows` | `{<window>: <glazed area m²>}`. |
| `state_to_gs` | Optional map from glazing state to glazing-system name, if they differ. |
| `pane_overrides` | Optional `{<layer index>: {"thickness_m", "density", "specific_heat", "conductivity"}}`. |
| `h_out`, `h_in` | Outdoor and indoor film coefficients [W/(m²K)], defaults 23 and 8; the indoor one is replaced by EnergyPlus' convection coefficient of the window at runtime. |
| `gap_nu` | Nusselt factor for the gas gaps, default 1. |
| `absorptance_mode` | `"hemispherical"` (the only mode). |

The entry reports the injected power [W]. See `frads_gym.cfs_thermal` in the [API reference](api.md).

## Normalisation

`observation_normalize` maps a raw value into the observation bounds `[low, high]` of
`observation_space`:

| Key | Meaning |
|---|---|
| `active` | Switch. |
| `dynamic` | `false`: static, from the range `low`–`high` below. `true`: from the percentiles of a moving window. `"ema"`: from exponentially tracked percentiles. |
| `low`, `high` | Static range of the raw value (lists for array values); also the starting range of the dynamic modes. |
| `percentile_margin` | Dynamic modes: the range is the `percentile_margin`-th to (100 - `percentile_margin`)-th percentile. Default 5. |
| `sample_length` | Window length in timesteps (`dynamic: true`). Default 8640. |
| `min_samples` | Warm-up in timesteps; until then the range blends from `low`–`high` to the running estimate. Default 96. |
| `ema_span` | Effective window in timesteps (`dynamic: "ema"`). Default 35040. |
| `clip`, `clip_mode` | `clip: false` disables clipping; `clip_mode` `"hard"` (default) clips, `"soft"` compresses with tanh. |
| `debug` | Print the range in use. |

`observation_standardize` turns the value into a z-score and maps `[low, high]` (in standard deviations,
default -3 to 3) to the observation bounds:

| Key | Meaning |
|---|---|
| `active` | Switch. Takes precedence over `observation_normalize`. |
| `dynamic` | `"welford"` (default): running mean and variance over everything seen. `"dynamic"`: over a moving window of `sample_length`. `"ema"`: exponential, with `ema_span`. |
| `low`, `high` | z-score bounds mapped to the observation bounds. |
| `max_deviation` | Clip the z-score to ±`max_deviation` first. |
| `std_threshold` | Floor for the standard deviation. |
| `min_samples`, `sample_length`, `ema_span` | As above. |
| `normalize_to_obs_space` | `false` returns the clipped z-score instead of mapping it. Default `true`. |
| `clip_mode` | `"hard"` or `"soft"`, as above. |
| `debug` | Print the statistics in use. |

The statistics of all dynamic modes persist across resets and can be saved and restored
(`FradsEnv.save_normalizer_state`, `normalizer_state_dir`).

## Environment variables

| Variable | Meaning |
|---|---|
| `FRADS_MATRIX_DIR` | Fixed matrix set, instead of `model_setup.fixed_matrices`. |
| `FRADS_MATRIX_WRITE=1` | Create the fixed matrix set instead of reading it. |
| `FRADS_NPROC` | Processes for generating Radiance matrices; default half the CPUs. |
| `FRADS_SENSOR_DEBUG=1` | Print which Radiance method computes each sensor. |

The variables of the [five-phase method](five-phase.md) need the frads fork.
