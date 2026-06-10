# GBOpt
GBOpt is a Python package for creating, manipulating, and optimizing bicrystal grain boundary structures through configurable global optimization workflows. It uses a modular architecture, with separate modules for structure creation, manipulation, and optimization. In the initial release:
- GB creation is facilitated through boundary-spec inputs that cover legacy five-degree-of-freedom parameters, exact P/Q orientation matrices, and CSL descriptions.
- GBs can be manipulated using grain translation, atom insertion, atom removal, and displacement along soft phonon modes.
- GB optimization is performed using either a Monte Carlo or an evolutionary algorithm optimizer, with energy evaluations currently using LAMMPS as the atomistic calculator.

The modular nature of the software is intended to allow easy extensibility to additional grain boundary parameterizations, structural manipulations, optimization engines, and external calculators.

## Boundary construction API

New code should build grain boundaries through `GBMaker.from_boundary_spec(...)`
rather than the deprecated direct `GBMaker(...)` constructor. Supported core
input formats are:

- `FiveDOFSpec(params)` for the legacy `[alpha, beta, gamma, theta, phi]`
  ZXZ-plus-inclination parameterization. `mode="exact"` rationalizes inputs
  that correspond to cubic CSL boundaries; use `mode="approximate"` for
  arbitrary or intentionally legacy-equivalent floating inputs.
- `PQSpec(P, Q)` for exact row-wise orientation matrices, where rows give
  the crystal directions aligned with lab x, y, and z for the left and right
  grains.
- `CSLExactSpec(axis, plane, quat, sigma=None)` and
  `CSLApproxSpec(axis, plane, angle_deg, sigma=None)` for CSL-oriented input.
  Exact CSL specs use an integer quaternion; approximate CSL specs use a
  floating-point angle.

Construction modes are:

- `mode="exact"` for exact integer P/Q construction. This requires `PQSpec`,
  `CSLExactSpec`, or a `FiveDOFSpec` that can be rationalized to a cubic CSL.
- `mode="prefer_exact"` to use exact construction when available and warn
  before falling back when exactification fails.
- `mode="approximate"` for the floating-point builder. This is the intended
  mode for incoherent or arbitrary-angle interfaces, not just a fallback.

Constructed objects expose `inplane_periodic` coherence metadata. Exact
coherent constructions report `(True, True)`, while approximate incoherent
interfaces report `(False, False)`.

Example:

```python
import numpy as np

from GBOpt import GBMaker
from GBOpt.BoundarySpec import FiveDOFSpec

gb = GBMaker.from_boundary_spec(
    3.52,
    "fcc",
    "Ni",
    FiveDOFSpec(np.array([0.643501, 0.0, 0.0, 0.0, -0.321751])),
    mode="approximate",
    gb_thickness=20.0,
)
```

To install, create a new conda environment
```
conda create --name GBOpt python
conda activate GBOpt
```

Make sure you are in the GBOpt directory:
```
cd ~/projects/GBOpt
```


Then install the dependencies
```
pip install .
```

## Citing GBOpt

If you use GBOpt in your research, please cite:

French, J. C. and Bhave, C. V. (2026). GBOpt: Grain Boundary Structure Optimization Using
Monte Carlo and Evolutionary Algorithms. *SoftwareX*, 35, 102763.
https://doi.org/10.1016/j.softx.2026.102763
