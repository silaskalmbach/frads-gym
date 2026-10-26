"""Optional five-phase method (5PM) wiring, switched on by FRADS_USE_FIVE_PHASE=1.

``EnergyPlusSetup`` defaults to ``radiance_method="3phase"``, and frads-gym creates it without that
argument. With the frads fork installed these patches let environment variables
select the five-phase method, its sun basis and the number of processes, so the same frads-gym code
runs either method. Without ``FRADS_USE_FIVE_PHASE`` everything stays on the three-phase default.

    FRADS_USE_FIVE_PHASE=1 FRADS_USE_ABSDF=1 SUN_BASIS=r2 [FRADS_NPROC=16]

Check with ``FRADS_SENSOR_DEBUG=1``: the log shows ``[CALLER_CALC_SENSOR] workflow_type=FivePhaseMethod``.
With upstream frads only the FRADS_NPROC patch is applied, and check_frads_capabilities() rejects the
fork-only variables.
"""

import inspect
import os
import warnings

_APPLIED = False
_FORK = False  # frads with the five-phase extensions (github.com/silaskalmbach/frads), set on apply

# Environment variables that change results but only exist in the frads fork; upstream frads
# would silently ignore them.
FORK_ENV_VARS = (
    "FRADS_USE_FIVE_PHASE", "SKY_SCALE", "SUN_SCALE", "FRADS_USE_ABSDF", "FRADS_USE_TT_T",
    "FRADS_ABSDF_XML_DIR", "FRADS_TT_XML_DIR", "FRADS_HYBRID_DIRECT", "FRADS_SKIP_ABSDF_SENSOR_SUN",
    "FRADS_BLACKED_OCTREE_MODIFIER", "FRADS_SKY_ROTATION", "FRADS_FASSADE",
)


def check_frads_capabilities(config: dict) -> None:
    """Raise if *config* or the environment needs features that only the frads fork provides."""
    if _FORK:
        return
    needs = [v for v in FORK_ENV_VARS if os.environ.get(v, "") not in ("", "0")]
    needs += [f"{k}: {s}" for k, v in config.items() if isinstance(v, dict)
              for s in ("calculate_dgp", "cfs_thermal", "cfs_thermal_systems") if s in v]
    if needs:
        import frads
        raise RuntimeError(
            f"The installed frads {getattr(frads, '__version__', '?')} lacks the extensions "
            f"needed by: {', '.join(needs)}. Remove these settings."
        )


def apply_five_phase_wiring() -> bool:
    """Patch ``frads.EnergyPlusSetup`` globally so that ``FRADS_USE_FIVE_PHASE=1`` selects
    ``radiance_method='5phase'`` with ``sun_basis`` (SUN_BASIS, default 'r2') and ``FRADS_NPROC``.
    Idempotent; returns True when the patches were applied by this call.
    """
    global _APPLIED
    if _APPLIED:
        return False
    try:
        import frads as _fr
        import frads.methods as _fm
    except ImportError as exc:  # pragma: no cover
        warnings.warn(f"five_phase_wiring: importing frads failed: {exc}")
        return False

    global _FORK
    _FORK = "radiance_method" in inspect.signature(_fr.EnergyPlusSetup.__init__).parameters

    # frads_wrapper calls initialize_radiance() without nproc, which would leave the matrix
    # generation single-threaded (frads default nproc=1); use FRADS_NPROC or half the CPUs.
    _orig_init_rad = _fr.EnergyPlusSetup.initialize_radiance

    def _init_rad_with_env(self, zones=None, nproc=None, view_matrices=False):
        if nproc is None:
            default_nproc = max(1, (os.cpu_count() or 4) // 2)
            nproc = int(os.environ.get("FRADS_NPROC", default_nproc))
        # view_matrices only exists in the fork; pass it only when requested
        kw = {"view_matrices": True} if view_matrices else {}
        return _orig_init_rad(self, zones=zones, nproc=nproc, **kw)

    _fr.EnergyPlusSetup.initialize_radiance = _init_rad_with_env

    _APPLIED = True
    if not _FORK:
        return True

    # Set sun_basis BEFORE FivePhaseMethod is constructed. EnergyPlusSetup.__init__ builds the
    # workflows, and FivePhaseMethod bakes sun-basis dependent state into them (direct_sun_matrix,
    # SunReceiver). Without this patch construction would use the settings default 'r6' before the
    # post-init loop below could set 'r2', and r6 arrays would end up under r2 cache keys.
    _orig_fm_init = _fm.FivePhaseMethod.__init__

    def _fm_init_with_env_sun_basis(self, config, *a, **kw):
        if os.environ.get("FRADS_USE_FIVE_PHASE") == "1":
            try:
                config.settings.sun_basis = os.environ.get("SUN_BASIS", "r2")
            except AttributeError:  # pragma: no cover
                pass
        _orig_fm_init(self, config, *a, **kw)

    _fm.FivePhaseMethod.__init__ = _fm_init_with_env_sun_basis

    _orig_eps_init = _fr.EnergyPlusSetup.__init__

    def _eps_init_with_env(self, *args, **kwargs):
        use_5pm = os.environ.get("FRADS_USE_FIVE_PHASE") == "1"
        if use_5pm:
            kwargs.setdefault("radiance_method", "5phase")
            # run initialize_radiance only after sun_basis is set
            user_init = kwargs.get("initialize_radiance", True)
            kwargs["initialize_radiance"] = False
        _orig_eps_init(self, *args, **kwargs)
        if use_5pm and getattr(self, "rconfigs", None):
            sun_basis = os.environ.get("SUN_BASIS", "r2")
            default_nproc = max(1, (os.cpu_count() or 4) // 2)
            nproc = int(os.environ.get("FRADS_NPROC", default_nproc))
            for cfg in self.rconfigs.values():
                cfg.settings.sun_basis = sun_basis
                cfg.settings.num_processors = nproc
            if user_init:
                # an nproc passed to EnergyPlusSetup explicitly wins over FRADS_NPROC
                self.initialize_radiance(nproc=kwargs.get("nproc", nproc))

    _fr.EnergyPlusSetup.__init__ = _eps_init_with_env

    # Optional scaling of the sky and sun contributions (default 1.0 = unchanged).
    _orig_calc_sensor = _fm.FivePhaseMethod.calculate_sensor

    def _calc_sensor_with_env_scales(self, sensor, bsdf, time, dni, dhi, **kwargs):
        kwargs.setdefault("sky_scale", float(os.environ.get("SKY_SCALE", "1.0")))
        kwargs.setdefault("sun_scale", float(os.environ.get("SUN_SCALE", "1.0")))
        return _orig_calc_sensor(self, sensor, bsdf, time, dni, dhi, **kwargs)

    _fm.FivePhaseMethod.calculate_sensor = _calc_sensor_with_env_scales

    # SKY_SCALE/SUN_SCALE must act on the image/DGP path as well: calculate_dgp takes Ev from the
    # (scaled) calculate_sensor but renders the HDR image with calculate_view. Without the same
    # scaling evalglare would combine scaled Ev with an unscaled image and bias the DGP.
    _orig_calc_view = _fm.FivePhaseMethod.calculate_view

    def _calc_view_with_env_scales(self, view, bsdf, time, dni, dhi, **kwargs):
        kwargs.setdefault("sky_scale", float(os.environ.get("SKY_SCALE", "1.0")))
        kwargs.setdefault("sun_scale", float(os.environ.get("SUN_SCALE", "1.0")))
        return _orig_calc_view(self, view, bsdf, time, dni, dhi, **kwargs)

    _fm.FivePhaseMethod.calculate_view = _calc_view_with_env_scales
    return True
