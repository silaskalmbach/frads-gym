"""Errors during setup and inside the EnergyPlus callback reach the caller instead of being swallowed."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from frads_gym.envs import frads_wrapper as fw


def _bare_simulation(**attrs):
    sim = object.__new__(fw.FradsSimulation)
    sim.__dict__.update({"epsetup_config": {}, "input_dir": None, "callback_error": None,
                         "simulation_finished": False, "obs_data_queue": queue.Queue(maxsize=1),
                         "action_data_queue": queue.Queue(maxsize=1), "next_step_event": threading.Event(),
                         **attrs})
    return sim


def test_callback_error_is_raised_in_main_thread(monkeypatch):
    stopped = []
    sim = _bare_simulation(epsetup=SimpleNamespace(
        api=SimpleNamespace(runtime=SimpleNamespace(stop_simulation=stopped.append))))
    monkeypatch.setattr(fw.context, "simulation", sim)
    monkeypatch.setattr(fw, "controller", lambda state: 1 / 0)

    fw.simplified_controller_wrapper("state")  # what EnergyPlus calls; must not raise into ctypes
    assert stopped == ["state"] and isinstance(sim.callback_error, ZeroDivisionError)
    fw.simplified_controller_wrapper("state")  # later callbacks return at once
    with pytest.raises(RuntimeError, match="EnergyPlus callback") as e:
        sim.steps()
    assert isinstance(e.value.__cause__, ZeroDivisionError)


def test_relative_path_needs_config_dir():
    with pytest.raises(ValueError, match="needs 'config_dir'"):
        _bare_simulation()._resolve("model.idf")


def test_missing_model_file_is_an_error(tmp_path):
    sim = _bare_simulation(input_dir=str(tmp_path),
                           epsetup_config={"model_setup": {"model_path": "missing.idf", "model_type": "medium_office"}})
    with pytest.raises(FileNotFoundError, match="model_path not found"):
        sim._model_setup()


def test_state_without_glazing_is_an_error():
    sim = _bare_simulation(epsetup_config={
        "model_setup": {"model_type": "small_office"},
        "facade_state_mapping": {"states": [{"name": "ec60", "max_threshold": 1.0}]}})
    with pytest.raises(ValueError, match=r"names \['ec60'\] match no glazing system"):
        sim._model_setup()


@pytest.mark.slow
@pytest.mark.simulation
def test_missing_energyplus_variable_fails_fast(tmp_path):
    """A variable this model does not have fails at the first step instead of reading 0 forever."""
    from pyenergyplus.dataset import weather_files

    from frads_gym import FradsEnv

    config = {"model_setup": {"model_type": "small_office"},
              "bogus": {"get_variable_value": {"name": "No Such Variable", "key": "Environment"},
                        "observation_space": {"low": 0, "high": 1}}}
    env = FradsEnv(output_dir=str(tmp_path), config_file=config, weather_files_path=[weather_files["usa_il_chicago"]],
                   reward_function=lambda *a: 0.0, enable_radiance=False,
                   run_period={"begin_month": 1, "begin_day": 2, "end_month": 1, "end_day": 2})
    try:
        with pytest.raises(RuntimeError, match="EnergyPlus callback.*'bogus'.*No Such Variable"):
            env.reset()
            env.step(np.array([0.0], dtype=np.float32))
    finally:
        env.close()
