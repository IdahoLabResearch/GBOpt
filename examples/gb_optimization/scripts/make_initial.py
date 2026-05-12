import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from GBOpt import GBMaker, GBManipulator

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def _load_boundaries():
    spec = importlib.util.spec_from_file_location(
        "boundaries", PROJECT_ROOT / "config" / "boundaries.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BOUNDARIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build initial GB structure")
    parser.add_argument("--material", required=True, help="Material name (e.g. Si)")
    parser.add_argument("--boundary", required=True, help="Boundary key from config/boundaries.py")
    args = parser.parse_args()

    mat_path = PROJECT_ROOT / "materials" / f"{args.material}.toml"
    with open(mat_path, "rb") as f:
        mat = tomllib.load(f)

    boundaries = _load_boundaries()
    if args.boundary not in boundaries:
        print(f"Unknown boundary '{args.boundary}'. Available: {list(boundaries)}", file=sys.stderr)
        sys.exit(1)

    bnd = boundaries[args.boundary]
    misorientation = bnd["misorientation"]
    repeat_factor = tuple(mat.get("repeat_factor", bnd["repeat_factor"]))

    a0 = mat["lattice_parameter"]
    structure = mat["structure"]
    atom_types = mat["atom_types"]
    r = mat["interaction_distance"]
    x_dim_min = mat.get("x_dim_min", 60)

    GB = GBMaker(
        a0, structure, a0, misorientation,
        atom_types=atom_types,
        interaction_distance=r,
        x_dim_min=x_dim_min,
        repeat_factor=repeat_factor,
        vacuum=0,
    )
    gb_thickness = 2 * max(GB.spacing["x"]["left"], GB.spacing["x"]["right"])

    GB = GBMaker(
        a0, structure, gb_thickness, misorientation,
        atom_types=atom_types,
        interaction_distance=r,
        x_dim_min=x_dim_min,
        repeat_factor=repeat_factor,
        vacuum=0,
    )

    manip = GBManipulator(GB)
    init_system = np.array(manip.parents[0].whole_system, copy=True)
    GB.write_lammps("initial.dat", init_system, manip.parents[0].box_dims, type_as_int=True)
    print(f"Written initial.dat  ({len(init_system)} atoms)")


if __name__ == "__main__":
    main()
