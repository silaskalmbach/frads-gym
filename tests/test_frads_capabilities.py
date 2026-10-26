"""Fork-only settings must fail loudly with upstream frads instead of being ignored."""

from __future__ import annotations

import pytest

from frads_gym import five_phase_wiring as w


@pytest.fixture
def upstream(monkeypatch):
    for v in w.FORK_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(w, "_FORK", False)
    return monkeypatch


def test_plain_config_passes(upstream):
    w.check_frads_capabilities({"wpi": {"calculate_wpi": {"zone": "z"}}})


@pytest.mark.parametrize("value", ["", "0"])
def test_unset_or_zero_env_var_passes(upstream, value):
    upstream.setenv("FRADS_USE_ABSDF", value)
    w.check_frads_capabilities({})


def test_fork_env_var_raises(upstream):
    upstream.setenv("FRADS_USE_FIVE_PHASE", "1")
    with pytest.raises(RuntimeError, match="FRADS_USE_FIVE_PHASE.*Remove these settings"):
        w.check_frads_capabilities({})


def test_calculate_dgp_raises(upstream):
    with pytest.raises(RuntimeError, match="glare: calculate_dgp"):
        w.check_frads_capabilities({"glare": {"calculate_dgp": {"zone": "z"}}})


def test_cfs_thermal_raises(upstream):
    with pytest.raises(RuntimeError, match="model_setup: cfs_thermal_systems, gain: cfs_thermal"):
        w.check_frads_capabilities({"model_setup": {"cfs_thermal_systems": []}, "gain": {"cfs_thermal": {}}})


def test_zero_scale_raises(upstream):
    upstream.setenv("SKY_SCALE", "0")  # removes the sky in the fork, so it is a setting, not "off"
    with pytest.raises(RuntimeError, match="SKY_SCALE"):
        w.check_frads_capabilities({})


def test_comment_entries_are_ignored(upstream):
    w.check_frads_capabilities({"_old": {"calculate_dgp": {"zone": "z"}}})


def test_fork_allows_everything(upstream):
    upstream.setattr(w, "_FORK", True)
    upstream.setenv("FRADS_USE_FIVE_PHASE", "1")
    w.check_frads_capabilities({"glare": {"calculate_dgp": {"zone": "z"}}})
