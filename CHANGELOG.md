# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-26

First release meant for use outside the original project. Simulation results for existing configurations
are unchanged unless noted under "Changed".

### Added

- Runs on frads from PyPI (`frads>=2.1.15,<2.2`); settings that this frads does not support raise an
  error instead of being ignored.
- Configuration and argument checks at construction (`frads_gym.validation`), listing every problem.
- Errors inside the EnergyPlus callback are raised in the main thread instead of being swallowed.
- Self-contained examples (`make_glazing.py`, `quickstart.py`, `energyplus_only.py`, `train_ppo.py`)
  without downloaded data.
- Documentation (MkDocs) with a configuration reference; CI on Linux, Windows and macOS with
  Python 3.10–3.12.
- Fixed Radiance matrix sets for bit-for-bit repeatable runs (`model_setup.fixed_matrices`,
  `FRADS_MATRIX_DIR`, `FRADS_MATRIX_WRITE=1` writes the set and its manifest).
- Per-sensor `calibration`, custom sensor and view positions.
- Wrappers: `ActionRepeatWrapper`, `ActiveHoursWrapper`, `SolarForecastWrapper`, `TemporalHistoryWrapper`.
- Options `sync_first_action`, `truncate_month_only`, `weekday_from_ep`, `staggered_start` with
  `env_id`/`n_envs`, `run_periods` per weather file, `normalizer_state_dir`; all off by default.
- EMA and windowed observation standardisation, soft clipping, normaliser state save and load.
- `info` exposes `raw_next_city_id`, `raw_next_step_idx`, `weather_file_idx` and calibration factors.

### Changed

- `output_dir`, `weather_files_path` and `reward_function` are required; there is no San Francisco
  weather default and no output in the package directory.
- No silent fallbacks: a missing `model_path` no longer falls back to `model_type`, missing glazing files,
  missing EnergyPlus variables, failed illuminance or glare calculations and unknown `ActiveHoursWrapper`
  keys raise instead of reporting 0 or −1.
- The action vector follows the config order of the `actuate_cfs_state` entries (it was sorted by name,
  while actuation used config order). Configurations with one controlled window are unaffected.
- `datetime` observation has six features (hour, day of year, day of week as sine and cosine).
- HVAC energy variables are summed over the HVAC sub-steps of a zone timestep.
- English messages and documentation throughout.
- `pyenergyplus-lbnl` is pinned to 25.2.0, the version frads requires.

### Removed

- `calculate_wpi` entries (use `calculate_illuminance` with `"post_processing": "mean"`).
- `normalizer_state_path` (use `normalizer_state_dir`).
- The bundled `pyenergyplus` wheels and the example data files.

### Fixed

- Deadlocks and hangs at EnergyPlus teardown and after step timeouts.
- Observation shapes with several parallel environments; element-wise standardisation of array
  observations.
- Several `calculate_edgps` entries with different views in one zone are an error; the last view used to
  replace the others silently.
- Entries whose key starts with `_` are ignored at runtime too, as documented.

## [0.1.1] - 2026-02-06

### Added
- Initial creating of `CHANGELOG.md`.
- Added `manual_release_guide.md` to documentation.

### Changed
- Bumped version to 0.1.1.
