# Quickstart

The examples in [`examples/`](https://github.com/silaskalmbach/frads-gym/tree/main/examples) run without
any downloaded data: the building model and the weather files ship with `pyenergyplus`, and the glazing
is generated.

```bash
git clone https://github.com/silaskalmbach/frads-gym.git
cd frads-gym/examples
python quickstart.py
```

| Script | What it shows |
|---|---|
| `make_glazing.py` | Four synthetic electrochromic states (double glazing, tintable outer pane), computed with pywincalc; about a minute, then cached |
| `quickstart.py` | One June day in the DOE medium office (Chicago) with Radiance; tints around noon and prints the work-plane illuminance |
| `energyplus_only.py` | The same day without Radiance, once clear and once dark, comparing the sensible cooling energy of the zone |
| `train_ppo.py` | The stable-baselines3 wiring for PPO (`pip install stable-baselines3`) |

All outputs go to `examples/output/`.

## The configuration

`quickstart.py` builds its configuration in Python; as JSON it reads:

```json
{
  "model_setup": {
    "model_type": "medium_office",
    "glazing_systems": [{"file": "output/glazing/ec60.json"}, {"file": "output/glazing/ec18.json"},
                        {"file": "output/glazing/ec06.json"}, {"file": "output/glazing/ec01.json"}]
  },
  "facade_state_mapping": {"states": [
    {"name": "ec60", "max_threshold": 0.25}, {"name": "ec18", "max_threshold": 0.5},
    {"name": "ec06", "max_threshold": 0.75}, {"name": "ec01", "max_threshold": 1.0}]},
  "cfs_state": {
    "get_cfs_state": {"key": "Perimeter_mid_ZN_1_Wall_South_Window"},
    "actuate_cfs_state": {"key": "Perimeter_mid_ZN_1_Wall_South_Window"},
    "observation_space": {"low": 0, "high": 1}, "action_space": {"low": 0, "high": 1}},
  "outdoor_temp": {
    "get_variable_value": {"name": "Site Outdoor Air Drybulb Temperature", "key": "Environment"},
    "observation_space": {"low": 0, "high": 1},
    "observation_normalize": {"active": true, "dynamic": false, "low": -20, "high": 40}},
  "wpi": {
    "calculate_illuminance": {"zone": "Perimeter_mid_ZN_1",
                              "cfs_name": ["Perimeter_mid_ZN_1_Wall_South_Window"]},
    "post_processing": "mean",
    "observation_space": {"low": 0, "high": 1},
    "observation_normalize": {"active": true, "dynamic": false, "low": 0, "high": 2000}}
}
```

(`indoor_temp`, `direct_normal_irradiance` and `diffuse_horizontal_irradiance` follow the pattern of
`outdoor_temp`.)

- `model_setup` loads the reference model and registers the four glazing systems as complex fenestration
  states.
- `facade_state_mapping` turns the action in [0, 1] into one of these states: up to 0.25 is `ec60`, up to
  0.5 `ec18`, and so on.
- `cfs_state` makes the south window of `Perimeter_mid_ZN_1` controllable and observes its state.
- `wpi` computes the work-plane illuminance on frads' default sensor grid with the three-phase method and
  averages it.

The [Configuration](configuration.md) page documents every key.

## The loop

```python
env = make_env("output/quickstart", run_period={"begin_year": 2002, "begin_month": 6, "begin_day": 21,
                                                "end_year": 2002, "end_month": 6, "end_day": 21})
obs, info = env.reset()
for step in range(10_000):
    hour = step / 4
    action = np.array([0.9 if 11 <= hour < 15 else 0.1], dtype=np.float32)   # tint around noon
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The observations in `obs` are normalised to [0, 1]; the raw values are in `info["raw_next_<key>"]`. The
first run creates the four glazing states and the Radiance matrices for every zone with windows; on a
32-core machine the whole script took just under two minutes, later runs reuse the glazing. The output
looks like this:

```text
 6.00 h  state=ec60  WPI=  804.3 lx  reward=-0.609
 8.00 h  state=ec60  WPI= 1631.8 lx  reward=-2.264
10.00 h  state=ec60  WPI= 3320.3 lx  reward=-5.641
12.00 h  state=ec01  WPI=   44.0 lx  reward=-0.912
14.00 h  state=ec01  WPI=   27.6 lx  reward=-0.945
16.00 h  state=ec60  WPI= 1076.6 lx  reward=-1.153
```

The last digits differ from run to run because Radiance samples the matrices randomly; a
[fixed matrix set](how-it-works.md#reproducibility) removes that.
