#!/usr/bin/env python
"""
find_min_gbe.py
Print the path and discovery point of the minimum-GBE structure from a
single MC or GA structure_opt run directory.

Usage:
    python scripts/find_min_gbe.py --material Fe --method GA --run run1
    python scripts/find_min_gbe.py --material Fe --method MC --run run3
    python scripts/find_min_gbe.py --material Fe --method GA --run run1 --init-type high_energy
    python scripts/find_min_gbe.py --material Fe --method MC --run run1 --e-accept 0.01
    python scripts/find_min_gbe.py --material Fe --method GA --run run1 --digits 4
    python scripts/find_min_gbe.py --material Fe --method MC --run run1 --digits 4

--digits truncates comparisons to N decimal places, so structures whose GBE
values agree at that precision are treated as ties; the earliest
generation/candidate (GA) or step (MC) wins.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PENALTY_THRESHOLD = 1.0e29
BOUNDARY_DEFAULT = "sigma5_310_STGB"


def _trunc(value: float, digits: int | None) -> float:
    """Return value rounded to *digits* decimal places, or unchanged if digits is None."""
    return round(value, digits) if digits is not None else value


# ---------------------------------------------------------------------------
# GA helpers
# ---------------------------------------------------------------------------

def _parse_ga_gbe_file(args):
    """
    Module-level worker (required for pickling).
    args: (gbe_file_str, digits)
    Returns (rounded_gbe, raw_gbe, gen_str, cand_str, dump_path_str) or None.
    """
    gbe_file_str, digits = args
    gbe_file = Path(gbe_file_str)
    try:
        parts = gbe_file.read_text().split()
        gbe = float(parts[1])
    except Exception:
        return None
    if gbe >= PENALTY_THRESHOLD:
        return None

    m = re.search(r"_g(\d+)_c(\d+)\.dump", gbe_file.name)
    if not m:
        return None
    gen, cand = m.group(1), m.group(2)

    dump = gbe_file.parent / f"final_GA_1_g{gen}_c{cand}.dump"
    if not dump.is_file():
        return None

    return _trunc(gbe, digits), gbe, gen, cand, str(dump)


def find_min_ga(run_dir: Path, digits: int | None = None, n_workers: int | None = None):
    """
    Return (gbe, generation, candidate, dump_path) for the minimum-GBE
    structure found in a single GA run directory.

    digits:    if given, GBE values are rounded to this many decimal places
               before comparison, so the earliest structure at that precision wins.
    n_workers: number of parallel worker processes (default: all CPUs).
    """
    workdir = run_dir / "workdir.1"
    if not workdir.is_dir():
        sys.exit(f"workdir.1 not found under {run_dir}")

    gbe_files = sorted(workdir.glob("gen_*/temp_GA_1_g*_c*.dump_GBE.txt"))
    if not gbe_files:
        sys.exit(f"No GBE files found under {workdir}")

    n = len(gbe_files)
    workers = n_workers or os.cpu_count()
    print(f"Scanning {n} GBE files across {workers} workers...", file=sys.stderr)

    work = [(str(f), digits) for f in gbe_files]

    best_trunc = float("inf")
    best_gbe = float("inf")
    best_gen = best_cand = best_dump = None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_parse_ga_gbe_file, work, chunksize=max(1, n // (workers * 4))):
            if result is None:
                continue
            trunc_gbe, raw_gbe, gen, cand, dump_str = result
            if trunc_gbe < best_trunc or (trunc_gbe == best_trunc and raw_gbe < best_gbe):
                best_trunc, best_gbe = trunc_gbe, raw_gbe
                best_gen, best_cand, best_dump = gen, cand, Path(dump_str)

    if best_dump is None:
        sys.exit("Could not locate any valid structure with a finite GBE.")

    return best_gbe, int(best_gen), int(best_cand), best_dump


# ---------------------------------------------------------------------------
# MC helpers
# ---------------------------------------------------------------------------

def find_min_mc(run_dir: Path, digits: int | None = None):
    """
    Return (gbe, step, dump_path) for the minimum-GBE structure found in a
    single MC run directory.  The step is the 0-based line index in
    gbe_vals.txt (i.e. MC step number).

    digits: if given, GBE values are rounded to this many decimal places
            before comparison, so the earliest step at that precision wins.
    """
    gbe_file = run_dir / "gbe_vals.txt"
    if not gbe_file.is_file():
        sys.exit(f"gbe_vals.txt not found in {run_dir}")

    best_gbe = float("inf")
    best_step = None

    for idx, line in enumerate(gbe_file.read_text().splitlines()):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            gbe = float(parts[1])
        except ValueError:
            continue
        if gbe < PENALTY_THRESHOLD and _trunc(gbe, digits) < _trunc(best_gbe, digits):
            best_gbe = gbe
            best_step = idx

    if best_step is None:
        sys.exit("No valid GBE values found in gbe_vals.txt.")

    for candidate in ("min_final_1.dump", "final_1.dump"):
        dump = run_dir / candidate
        if dump.is_file():
            return best_gbe, best_step, dump

    sys.exit(f"Minimum GBE found at step {best_step} but no dump file located in {run_dir}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print the minimum-GBE structure path and the "
                    "generation/step at which it was found, for a single run."
    )
    parser.add_argument("--material", required=True)
    parser.add_argument("--boundary", default=BOUNDARY_DEFAULT)
    parser.add_argument("--method", required=True, choices=["GA", "MC"],
                        help="Optimizer type")
    parser.add_argument("--run", required=True,
                        help="Run directory name, e.g. run1")
    parser.add_argument("--init-type", default="standard",
                        choices=["standard", "high_energy"],
                        help="Initialization strategy (default: standard)")
    parser.add_argument("--e-accept", default="0.1",
                        help="MC acceptance threshold string used in directory name "
                             "(default: 0.1, only used when --method MC)")
    parser.add_argument("--digits", type=int, default=None,
                        help="Round GBE values to this many decimal places before "
                             "comparing, so the earliest structure at that precision "
                             "wins (default: full float precision)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes for GA file scanning "
                             "(default: all CPUs, only used when --method GA)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.method == "GA":
        base_dir = (
            project_root / args.material / args.boundary
            / "GA" / args.init_type
        )
        run_dir = base_dir / args.run
        if not run_dir.is_dir():
            sys.exit(f"Run directory not found: {run_dir}")

        gbe, gen, cand, dump = find_min_ga(run_dir, digits=args.digits, n_workers=args.workers)
        print(f"Minimum GBE : {gbe:.6f} J/m²"
              + (f"  (compared at {args.digits} d.p.)" if args.digits is not None else ""))
        print(f"Found at    : generation {gen}, candidate {cand}")
        print(f"Structure   : {dump}")

    else:  # MC
        mc_dir = (f"standard_{args.e_accept}" if args.init_type == "standard"
                  else "high_energy_0.1")
        base_dir = (
            project_root / args.material / args.boundary
            / "MC" / mc_dir
        )
        run_dir = base_dir / args.run
        if not run_dir.is_dir():
            sys.exit(f"Run directory not found: {run_dir}")

        gbe, step, dump = find_min_mc(run_dir, digits=args.digits)
        print(f"Minimum GBE : {gbe:.6f} J/m²"
              + (f"  (compared at {args.digits} d.p.)" if args.digits is not None else ""))
        print(f"Found at    : step {step}")
        print(f"Structure   : {dump}")


if __name__ == "__main__":
    main()
