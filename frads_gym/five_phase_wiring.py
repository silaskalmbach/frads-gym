"""Optionales 5-Phase-Wiring fuer das RL-Training (gated durch FRADS_USE_FIVE_PHASE).

Spiegelt den bewaehrten Monkeypatch aus
``rl/evaluation/run_ftg_calibration_sim.py`` (Zeile 51-115), damit der RL-Pfad
(``main_optimize`` -> ``frads_wrapper`` / ``matrices_manager``) DENSELBEN
5PM-Zustand nutzt wie die FTG-Kalibrierung (X22-Winning-Pipeline). Hintergrund:
``EnergyPlusSetup`` hat Default ``radiance_method="3phase"``; die per-Sensor-k
und k=1.45 wurden aber gegen 5PM-erzeugte Ev/WPI gefittet. Ohne dieses Wiring
liefe das Training auf 3PM -> Methodenbruch zur Kalibrierung (2026-06-04).

**Gated:** Ohne ``FRADS_USE_FIVE_PHASE`` bleibt alles beim Default 3phase
(medium_office unberuehrt). Aktivierung (FTG, X22-konform):

    FRADS_USE_FIVE_PHASE=1 FRADS_USE_ABSDF=1 SUN_BASIS=r2 [FRADS_NPROC=16]

Verifizieren via Lauf-Log ``[CALLER_CALC_SENSOR] workflow_type=FivePhaseMethod``.
"""

import os
import warnings

_APPLIED = False


def apply_five_phase_wiring() -> bool:
    """Patcht ``frads.EnergyPlusSetup`` global, sodass ``FRADS_USE_FIVE_PHASE=1``
    ``radiance_method='5phase'`` + ``sun_basis`` (SUN_BASIS, default 'r2') +
    ``FRADS_NPROC`` aktiviert. Idempotent (mehrfacher Aufruf = einmal gepatcht).
    Gibt True zurueck, wenn (neu) angewandt.
    """
    global _APPLIED
    if _APPLIED:
        return False
    try:
        import frads as _fr
        import frads.methods as _fm
    except ImportError as exc:  # pragma: no cover
        warnings.warn(f"five_phase_wiring: frads-Import fehlgeschlagen: {exc}")
        return False

    # sun_basis VOR der FivePhaseMethod-Konstruktion in die Config injizieren.
    # eplus.py baut die Workflows (FivePhaseMethod.__init__) INNERHALB von
    # EnergyPlusSetup.__init__; FivePhaseMethod backt dort sun_basis-abhaengigen
    # Zustand ein (direct_sun_matrix, SunReceiver). Ohne diesen Patch liefe die
    # Konstruktion auf dem Settings-Default 'r6', bevor die Post-Init-Schleife
    # unten sun_bas!='r2' setzen koennte -> r6-Arrays landen unter r2-Cache-Keys.
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
            # initialize_radiance erst nach Setzen der sun_basis ausfuehren.
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
                # Ein explizit an EnergyPlusSetup uebergebenes nproc-kwarg hat
                # Vorrang vor FRADS_NPROC/CPU-Default (Spiegel zu
                # run_ftg_calibration_sim.py: kwargs.get('nproc', nproc)).
                self.initialize_radiance(nproc=kwargs.get("nproc", nproc))

    _fr.EnergyPlusSetup.__init__ = _eps_init_with_env

    # frads_wrapper ruft initialize_radiance() ohne nproc -> eplus.py-Default
    # nproc=1 (rcontrib single-thread). FRADS_NPROC honorieren.
    _orig_init_rad = _fr.EnergyPlusSetup.initialize_radiance

    def _init_rad_with_env(self, zones=None, nproc=None, view_matrices=False):
        if nproc is None:
            default_nproc = max(1, (os.cpu_count() or 4) // 2)
            nproc = int(os.environ.get("FRADS_NPROC", default_nproc))
        return _orig_init_rad(self, zones=zones, nproc=nproc, view_matrices=view_matrices)

    _fr.EnergyPlusSetup.initialize_radiance = _init_rad_with_env

    # Variante-A Pfad-Skalierung (Default 1.0 = neutral, = X22-Zustand).
    _orig_calc_sensor = _fm.FivePhaseMethod.calculate_sensor

    def _calc_sensor_with_env_scales(self, sensor, bsdf, time, dni, dhi, **kwargs):
        kwargs.setdefault("sky_scale", float(os.environ.get("SKY_SCALE", "1.0")))
        kwargs.setdefault("sun_scale", float(os.environ.get("SUN_SCALE", "1.0")))
        return _orig_calc_sensor(self, sensor, bsdf, time, dni, dhi, **kwargs)

    _fm.FivePhaseMethod.calculate_sensor = _calc_sensor_with_env_scales

    # SKY_SCALE/SUN_SCALE MUESSEN konsistent auch auf den Bild-/DGP-Pfad wirken:
    # calculate_dgp holt Ev ueber das (skalierte) calculate_sensor, rendert das
    # HDR aber ueber calculate_view. Ohne denselben Scale-Patch bekaeme evalglare
    # skaliertes Ev zu unskaliertem HDR -> systematisch verzerrte DGP. calculate_view
    # traegt dieselben sky_scale/sun_scale-Parameter wie calculate_sensor.
    _orig_calc_view = _fm.FivePhaseMethod.calculate_view

    def _calc_view_with_env_scales(self, view, bsdf, time, dni, dhi, **kwargs):
        kwargs.setdefault("sky_scale", float(os.environ.get("SKY_SCALE", "1.0")))
        kwargs.setdefault("sun_scale", float(os.environ.get("SUN_SCALE", "1.0")))
        return _orig_calc_view(self, view, bsdf, time, dni, dhi, **kwargs)

    _fm.FivePhaseMethod.calculate_view = _calc_view_with_env_scales

    _APPLIED = True
    return True
