# API reference

## Environment

::: frads_gym.FradsEnv
    options:
      members: [__init__, reset, step, close, save_normalizer_state, load_normalizer_state]

## Wrappers

::: frads_gym.wrappers.ActionRepeatWrapper

::: frads_gym.wrappers.ActiveHoursWrapper

::: frads_gym.wrappers.SolarForecastWrapper

::: frads_gym.wrappers.TemporalHistoryWrapper

## Configuration checks

::: frads_gym.validation
    options:
      members: [ConfigError, validate_config, validate_env_args]

## Transient glazing model

::: frads_gym.cfs_thermal
    options:
      members: [CFSThermalModel, add_other_equipment, PaneThermalProps, GapThermalProps, StepResult]

## frads capabilities

::: frads_gym.five_phase_wiring
    options:
      members: [check_frads_capabilities, FORK_ENV_VARS]
