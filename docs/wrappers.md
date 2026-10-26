# Wrappers

`frads_gym.wrappers` holds four Gymnasium wrappers for training. They can be stacked; a typical order is

```python
from frads_gym.wrappers import (ActionRepeatWrapper, ActiveHoursWrapper, SolarForecastWrapper,
                                TemporalHistoryWrapper)

env = ActionRepeatWrapper(env, repeat=2)
env = ActiveHoursWrapper(env, check_solar=True)
env = SolarForecastWrapper(env, forecast_horizon=4)
env = TemporalHistoryWrapper(env, max_time_steps=4)
```

## ActionRepeatWrapper

Holds each action for `repeat` consecutive timesteps and returns the sum of their rewards, so the agent
decides less often than the simulation steps. `environment_log.csv` still has one row per timestep.

## ActiveHoursWrapper

Skips timesteps in which the facade has no influence, so the agent only sees active transitions:

- `check_solar=True`: active when the sum of `solar_keys` (default `direct_normal_irradiance` and
  `diffuse_horizontal_irradiance`) is above 0.
- `check_occupancy=True`: active when `occupancy_key` (default `occupant_count_1`) is above 0.
- `mode`: `"or"` (default) or `"and"` when both checks are on.
- `inactive_action`: the action applied while skipping; by default the last action is held.

The keys refer to configuration entries (their `raw_next_<key>` values in `info`); a missing key is an
error. Skipped timesteps are still simulated and logged, and the reward function is not called for them
(`info["agent_active"]` is false).

## SolarForecastWrapper

Adds the global horizontal irradiance of the next `forecast_horizon` timesteps, read from the weather
file of the running episode and divided by `ghi_max` (default 1200 W/m²), as `info["solar_forecast"]` and,
with `add_to_obs=True`, as the observation key `solar_forecast`. The forecast is perfect (taken from the
EPW); the observation right after `reset()` is zero.

## TemporalHistoryWrapper

Stacks the last `max_time_steps` observations for sequence models: `obs_history` with shape
`(max_time_steps, flat_obs_dim)` and `temporal_mask` (1 for valid, 0 for padding). `flatten_keys` limits
the keys that go into the history.

See the [API reference](api.md) for all parameters.
