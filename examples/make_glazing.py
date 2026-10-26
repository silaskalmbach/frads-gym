"""Create four synthetic electrochromic glazing states, without any downloaded data.

Each state is a double glazing: a tintable outer pane with a flat spectrum and a clear inner
pane, computed with pywincalc through frads. The values are illustrative, not a real product.

    python make_glazing.py [out_dir]
"""

import json
import sys
from pathlib import Path

import frads as fr
import numpy as np

# state name -> solar and visible transmittance of the tintable pane (clear ... dark)
STATES = {"ec60": 0.60, "ec18": 0.18, "ec06": 0.06, "ec01": 0.01}


def pane(name, transmittance, reflectance, thickness_mm=6.0):
    """A specular pane with the same transmittance and reflectance at every wavelength."""
    wavelengths = np.round(np.arange(0.30, 2.501, 0.01), 3)  # 300-2500 nm in micrometres
    return json.dumps({
        "product_id": None, "name": name, "type": "glazing", "subtype": "monolithic", "coated_side": "Neither",
        "manufacturer_name": "synthetic",
        "measured_data": {"is_specular": True, "thickness": thickness_mm, "tir_front": 0.0,
                          "emissivity_front": 0.84, "emissivity_back": 0.84, "conductivity": 1.0},
        "spectral_data": {"spectral_data": [
            {"wavelength": float(w), "T": transmittance, "Rf": reflectance, "Rb": reflectance}
            for w in wavelengths]},
    }).encode()


def make_glazing(out_dir):
    """Write one frads glazing-system JSON per state to *out_dir* (skips existing files)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, t in STATES.items():
        path = out_dir / f"{name}.json"
        if not path.exists():
            gs = fr.create_glazing_system(name, [fr.LayerInput(pane(f"{name}_pane", t, 0.10)),
                                                 fr.LayerInput(pane("clear_pane", 0.85, 0.08))])
            gs.save(path)
            print(f"wrote {path}")
        paths.append(path)
    return paths


if __name__ == "__main__":
    make_glazing(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "output" / "glazing")
