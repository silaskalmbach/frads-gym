# Installation

```bash
pip install frads-gym
```

This installs frads-gym together with everything it runs on:

| Package | Provides |
|---|---|
| `frads` | EnergyPlus model handling, complex fenestration systems, Radiance matrix workflows |
| `pyenergyplus-lbnl` | the EnergyPlus 25.2 runtime and its Python API, plus reference models and weather files |
| `pyradiance` | the Radiance binaries |
| `pywincalc` | optical calculations for glazing systems |
| `gymnasium`, `numpy` | the environment API and arrays |

No engine has to be installed separately.

The EnergyPlus version is fixed: frads requires `pyenergyplus-lbnl==25.2.0`, so EnergyPlus 25.2 is used
even where newer versions exist. Building models must be EnergyPlus 25.2 IDF or epJSON files; frads converts
the file format, not the version. Convert older models with the IDFVersionUpdater of a full EnergyPlus
installation (it is not part of `pyenergyplus-lbnl`).

## Supported platforms

Python 3.10, 3.11 and 3.12 on Linux (x86_64), Windows (x86_64) and macOS (arm64). The unit tests run on
all nine combinations, the simulation tests (EnergyPlus and Radiance) on Linux. Python 3.13 is not supported yet because `pywincalc` publishes no wheels for
it.

To check the installation:

```bash
python -c "import frads_gym; print(frads_gym.FradsEnv)"
```

Then run the [Quickstart](quickstart.md).

## Development install

```bash
git clone https://github.com/silaskalmbach/frads-gym.git
cd frads-gym
pip install -e ".[test]"
python -m pytest -m "not slow" tests      # a few seconds
python -m pytest -m slow tests            # runs EnergyPlus and Radiance, a few minutes
```

See [Contributing](contributing.md) for the rest of the workflow.
