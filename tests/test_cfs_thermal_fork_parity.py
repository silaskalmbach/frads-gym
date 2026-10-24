"""frads_gym.cfs_thermal must behave exactly like the frads fork it was moved from.

Runs only where the fork (``frads.cfs_thermal``) is installed; elsewhere it is skipped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

fork = pytest.importorskip("frads.cfs_thermal")

import frads as fr  # noqa: E402
from pyenergyplus.dataset import ref_models  # noqa: E402

from frads_gym import cfs_thermal as ours  # noqa: E402


def _glazing(abs_outer):
    layer = lambda t, k: SimpleNamespace(thickness_m=t, conductivity=k, emissivity_front=0.84,  # noqa: E731
                                         emissivity_back=0.84, ir_transmittance=0.0)
    gas = [SimpleNamespace(gas="air", ratio=0.1), SimpleNamespace(gas="argon", ratio=0.9)]
    return SimpleNamespace(
        layers=[layer(0.006, 1.0), layer(0.004, None)],
        gaps=[SimpleNamespace(thickness_m=0.016, gas=gas)],
        solar_front_absorptance=[abs_outer, 0.03],
    )


def _run(mod):
    clear = [0.10 + 0.001 * i for i in range(145)]  # Klems-full vector
    tinted = [0.55 + 0.001 * i for i in range(145)]
    model = mod.CFSThermalModel.from_glazing_system(
        _glazing(clear), overrides={"1": {"density": 2600.0}, 0: {"thickness_m": 0.008}}, init_temp_c=21.0)
    out = []
    for k in range(288):  # three days at 15 min
        h = (k % 96) / 4
        incident = max(0.0, 800.0 * (1 - abs(h - 12) / 6))
        if k == 100:
            model.set_absorptance(tinted)
        r = model.step(incident, 5.0 + h / 2, 22.0, 900.0, h_in=3.0 + (k % 5))
        out.append((r.pane_temps_c, r.inward_transient_wm2, r.inward_steady_wm2, r.lag_correction_wm2))
    return out


def test_cfs_thermal_model_matches_fork():
    assert _run(ours) == _run(fork)


def test_add_other_equipment_matches_fork():
    a = fr.load_energyplus_model(ref_models["medium_office"])
    b = fr.load_energyplus_model(ref_models["medium_office"])
    zone = sorted(a.zone)[0]
    for replace in (False, True):  # second call replaces the first object
        a.add_other_equipment("cfs_gain", zone, fraction_radiant=0.6, replace=replace)
        ours.add_other_equipment(b, "cfs_gain", zone, fraction_radiant=0.6, replace=replace)
    assert a.model_dump_json() == b.model_dump_json()
    with pytest.raises(ValueError, match="already exists"):
        ours.add_other_equipment(b, "cfs_gain", zone)
    with pytest.raises(ValueError, match="not found"):
        ours.add_other_equipment(b, "x", "no_such_zone")
