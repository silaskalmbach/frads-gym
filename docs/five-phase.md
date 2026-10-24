# Five-phase method (frads fork)

The released frads computes daylight with the three-phase method, which describes the direct sun through
the coarse Klems basis of the glazing's BSDF. The five-phase method treats the direct sun separately and
is more accurate for direct sunlight on a sensor or in the view. It, image-based DGP and a few related
options are developed in a fork of frads
([silaskalmbach/frads](https://github.com/silaskalmbach/frads)) and are not part of the PyPI release.

Install the fork as described in [Installation](installation.md#five-phase-method-and-image-based-dgp-frads-fork).
frads-gym checks the installed frads: with upstream frads, each setting below raises an error that names
it.

## Configuration

- [`calculate_dgp`](configuration.md#calculate_dgp): image-based DGP and Ev from a rendered fisheye image
  (evalglare). frads-gym then also generates the view matrices.

## Environment variables

| Variable | Meaning |
|---|---|
| `FRADS_USE_FIVE_PHASE=1` | Use the five-phase method for all zones. |
| `SUN_BASIS` | Sun basis of the five-phase method, default `r2`. |
| `SKY_SCALE`, `SUN_SCALE` | Scale the sky and sun contributions of sensors and images (default 1). |
| `FRADS_USE_ABSDF=1`, `FRADS_ABSDF_XML_DIR` | Model the glazing as an aBSDF material from per-state Klems files `<state>_klems.xml` in that directory. |
| `FRADS_USE_TT_T=1`, `FRADS_TT_XML_DIR` | Use tensor-tree BSDF files `<state>_tt.xml` for the aBSDF material. |
| `FRADS_HYBRID_DIRECT=1` | Compute the direct-sun part of sensor values by ray tracing instead of with the sun matrix. |
| `FRADS_BLACKED_OCTREE_MODIFIER` | Material of the window in the blacked-out octree, `black` (default) or `glowing`. |
| `FRADS_SKIP_ABSDF_SENSOR_SUN=1` | Skip generating the sensor sun matrices (for image-only use). |
| `FRADS_SKY_ROTATION` | Rotate the sky by this many degrees, counter-clockwise seen from above. |
| `FRADS_FASSADE` | JSON overriding the extrusion of facade reveals per wall. |

These options are experimental and documented in the fork; frads-gym only passes them through.
