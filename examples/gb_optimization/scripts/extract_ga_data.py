#!/usr/bin/env python3
"""
extract_ga_data.py
------------------
Extract essential plotting data from pickled GA objects and save as
lightweight JSON files. Run this once on the full simulation directory
before committing to the examples repo.

Usage:
    python scripts/extract_ga_data.py
    python scripts/extract_ga_data.py --root /path/to/project
    python scripts/extract_ga_data.py --dry-run   # show what would be written
"""

import argparse
import glob
import json
import os
import pickle
import sys
import types
from pathlib import Path

PENALTY_THRESHOLD = 1.0e29


def _inject_ga_dummies():
    import __main__

    def _dummy(*a, **kw):
        pass

    __main__.get_gb_energy  = _dummy
    __main__.evaluate_batch = _dummy

    if "GBOpt" not in sys.modules:
        pkg = types.ModuleType("GBOpt")
        pkg.__path__ = []
        sys.modules["GBOpt"] = pkg

    if "GBOpt.GBMinimizer" not in sys.modules:
        class _AutoStub(types.ModuleType):
            def __getattr__(self, name):
                cls = type(name, (), {})
                setattr(self, name, cls)
                return cls
        mod = _AutoStub("GBOpt.GBMinimizer")
        sys.modules["GBOpt"].GBMinimizer = mod
        sys.modules["GBOpt.GBMinimizer"] = mod

    import GBOpt.GBMinimizer as _m
    if not hasattr(_m, "GeneticAlgorithmMinimizer"):
        _m.GeneticAlgorithmMinimizer = type("GeneticAlgorithmMinimizer", (), {})
    _m.GeneticAlgorithmMinimizer.gb_energy_func       = _dummy
    _m.GeneticAlgorithmMinimizer.gb_batch_energy_func = _dummy


def extract(pkl_path, dry_run=False):
    out_path = os.path.join(os.path.dirname(pkl_path), "ga_data.json")

    try:
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as exc:
        print(f"  SKIP (load error): {pkl_path}\n    {exc}")
        return None, None

    # Strip penalty values and empty generations
    gbe_vals = [
        [e for e in (g if hasattr(g, "__iter__") else [g])
         if e < PENALTY_THRESHOLD]
        for g in obj.GBE_vals
    ]
    gbe_vals = [g for g in gbe_vals if g]

    # Extract population_size — fall back to size of first generation
    pop_size = getattr(obj, "population_size", None)
    if pop_size is None and gbe_vals:
        pop_size = len(gbe_vals[0])

    data = {
        "population_size": int(pop_size) if pop_size is not None else None,
        "n_generations":   len(gbe_vals),
        "gbe_vals":        gbe_vals,
    }

    pkl_kb  = os.path.getsize(pkl_path) / 1024
    json_kb = len(json.dumps(data)) / 1024

    if dry_run:
        print(f"  would write: {out_path}")
        print(f"    pkl={pkl_kb:.0f} KB  ->  json≈{json_kb:.0f} KB  "
              f"({100*json_kb/pkl_kb:.0f}% of original)")
        return pkl_kb, json_kb

    with open(out_path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))  # compact, no whitespace

    print(f"  {out_path}")
    print(f"    pkl={pkl_kb:.0f} KB  ->  json={os.path.getsize(out_path)/1024:.0f} KB  "
          f"({100*os.path.getsize(out_path)/os.path.getsize(pkl_path):.0f}% of original)")

    return pkl_kb, os.path.getsize(out_path) / 1024


def parse_args():
    p = argparse.ArgumentParser(description="Extract GA plotting data from pkl files")
    p.add_argument("--root", default=None,
                   help="Project root (default: directory containing scripts/)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be written without writing anything")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root if args.root else str(Path(__file__).parent.parent)

    _inject_ga_dummies()

    pattern = os.path.join(root, "**", "pickled_ga_object.pkl")
    pkl_files = sorted(glob.glob(pattern, recursive=True))

    if not pkl_files:
        sys.exit(f"No pickled_ga_object.pkl files found under {root}")

    print(f"Found {len(pkl_files)} pkl file(s)"
          + (" — dry run" if args.dry_run else ""))

    total_pkl_kb = total_json_kb = 0.0
    for pkl_path in pkl_files:
        pkl_kb, json_kb = extract(pkl_path, dry_run=args.dry_run)
        if pkl_kb is not None:
            total_pkl_kb  += pkl_kb
            total_json_kb += json_kb

    print(f"\nTotal: {total_pkl_kb/1024:.1f} MB pkl  ->  "
          f"{total_json_kb/1024:.1f} MB json  "
          f"({100*total_json_kb/total_pkl_kb:.0f}% of original)")

    if not args.dry_run:
        print("\nAdd to .gitignore:")
        print("  pickled_ga_object.pkl")


if __name__ == "__main__":
    main()
