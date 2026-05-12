#!/usr/bin/env python3
"""
extract_ga_data.py
------------------
Extract essential plotting data from pickled GA objects and save as
lightweight JSON files. Run this once on the full simulation directory
before committing to the examples repo.

Each output ga_data.json contains:
    population_size  : int
    n_generations    : int
    gbe_vals         : list[list[float]]   – per generation, penalty-stripped
    atom_counts      : list[list[int]]     – matching shape, raw N_total per candidate

The atom_counts field is what plot_density_progress.py needs to compute
    [n] = (N_total mod N_bulk_plane) / N_bulk_plane
at plot time, rather than baking in [n] values here.

Usage:
    python scripts/extract_ga_data.py
    python scripts/extract_ga_data.py --root /path/to/project
    python scripts/extract_ga_data.py --dry-run   # show what would be written

NOTE: Adjust ATOM_COUNT_ATTRS below if your GA class uses a different
attribute name for per-candidate atom counts.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PENALTY_THRESHOLD = 1.0e29

# Candidate attribute names for per-candidate atom counts, tried in order.
# The first one found on the GA object is used.
# Each entry is either:
#   - a string  → obj.<name>  is expected to be list[list[int]]
#   - a tuple   → obj.<outer>.<inner>  (one level of nesting)
# Add or reorder entries to match your GA class.
ATOM_COUNT_ATTRS = [
    "atom_counts",      # e.g. obj.atom_counts
    "N_atoms",          # e.g. obj.N_atoms
    "natoms",           # e.g. obj.natoms
    "n_atoms",          # e.g. obj.n_atoms
    "total_atoms",      # e.g. obj.total_atoms
]


# ---------------------------------------------------------------------------
# Pickle loading helpers
# ---------------------------------------------------------------------------

def _inject_ga_dummies():
    """Stub out GA-internal imports so pickle.load works without the codebase."""
    import __main__

    def _dummy(*a, **kw):
        pass

    __main__.get_gb_energy = _dummy
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
    _m.GeneticAlgorithmMinimizer.gb_energy_func = _dummy
    _m.GeneticAlgorithmMinimizer.gb_batch_energy_func = _dummy


# ---------------------------------------------------------------------------
# Atom-count extraction
# ---------------------------------------------------------------------------

def _find_atom_counts(obj) -> list[list[int]] | None:
    """
    Try each name in ATOM_COUNT_ATTRS and return the first match as a
    list[list[int]] aligned with GBE_vals, or None if nothing is found.

    Prints the attribute name used so it is easy to confirm (or update
    ATOM_COUNT_ATTRS if the wrong one is picked).
    """
    for attr in ATOM_COUNT_ATTRS:
        if "." in attr:
            outer, inner = attr.split(".", 1)
            candidate = getattr(getattr(obj, outer, None), inner, None)
        else:
            candidate = getattr(obj, attr, None)

        if candidate is not None:
            print(f"    atom counts: using obj.{attr}")
            # Normalise to list[list[int]]
            try:
                return [[int(x) for x in gen] for gen in candidate]
            except (TypeError, ValueError) as exc:
                print(f"    WARNING: obj.{attr} found but could not be "
                      f"converted to list[list[int]]: {exc}")
                continue

    # Nothing found — emit a helpful diagnostic
    available = [a for a in dir(obj) if not a.startswith("__")]
    print(
        "    WARNING: no atom-count attribute found.\n"
        "    Available attributes on the GA object:\n"
        f"      {', '.join(available)}\n"
        "    Add the correct name to ATOM_COUNT_ATTRS at the top of this script."
    )
    return None


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def extract(pkl_path: str, dry_run: bool = False):
    out_path = os.path.join(os.path.dirname(pkl_path), "ga_data.json")

    try:
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as exc:
        print(f"  SKIP (load error): {pkl_path}\n    {exc}")
        return None, None

    # ── GBE values ──────────────────────────────────────────────────────────
    # Strip penalty values; keep parallel index structure for atom_counts.
    raw_gbe = obj.GBE_vals
    gbe_vals: list[list[float]] = []
    keep_mask: list[list[bool]] = []   # True = kept candidate

    for gen in raw_gbe:
        gen_list = list(gen) if hasattr(gen, "__iter__") else [gen]
        mask = [e < PENALTY_THRESHOLD for e in gen_list]
        gbe_vals.append([e for e, m in zip(gen_list, mask) if m])
        keep_mask.append(mask)

    # Drop entirely empty generations
    non_empty = [(g, m) for g, m in zip(gbe_vals, keep_mask) if g]
    if not non_empty:
        print(f"  SKIP (no valid GBE values): {pkl_path}")
        return None, None
    gbe_vals, keep_mask = map(list, zip(*non_empty))

    # ── Atom counts ─────────────────────────────────────────────────────────
    raw_counts = _find_atom_counts(obj)
    atom_counts: list[list[int]] | None = None

    if raw_counts is not None:
        # Apply the same penalty mask and empty-generation filter
        filtered = []
        for gen_counts, mask in zip(raw_counts, keep_mask):
            filtered.append([c for c, m in zip(gen_counts, mask) if m])
        atom_counts = [g for g in filtered if g]

        # Sanity-check shape alignment
        if len(atom_counts) != len(gbe_vals) or any(
            len(a) != len(g) for a, g in zip(atom_counts, gbe_vals)
        ):
            print(
                "    WARNING: atom_counts shape does not match gbe_vals after "
                "filtering — dropping atom_counts to avoid silent misalignment."
            )
            atom_counts = None

    # ── Population size ─────────────────────────────────────────────────────
    pop_size = getattr(obj, "population_size", None)
    if pop_size is None and gbe_vals:
        pop_size = len(gbe_vals[0])

    # ── Assemble output ─────────────────────────────────────────────────────
    data = {
        "population_size": int(pop_size) if pop_size is not None else None,
        "n_generations":   len(gbe_vals),
        "gbe_vals":        gbe_vals,
        "atom_counts":     atom_counts,   # None if not found; plot script handles this
    }

    pkl_kb = os.path.getsize(pkl_path) / 1024
    json_kb = len(json.dumps(data)) / 1024

    if dry_run:
        print(f"  would write: {out_path}")
        print(f"    pkl={pkl_kb:.0f} KB  ->  json≈{json_kb:.0f} KB  "
              f"({100 * json_kb / pkl_kb:.0f}% of original)")
        print(f"    atom_counts present: {atom_counts is not None}")
        return pkl_kb, json_kb

    with open(out_path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))

    final_kb = os.path.getsize(out_path) / 1024
    print(f"  {out_path}")
    print(f"    pkl={pkl_kb:.0f} KB  ->  json={final_kb:.0f} KB  "
          f"({100 * final_kb / pkl_kb:.0f}% of original)")
    print(f"    atom_counts present: {atom_counts is not None}")

    return pkl_kb, final_kb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract GA plotting data from pkl files")
    p.add_argument("--root", default=None,
                   help="Project root (default: $PROJECT_ROOT)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be written without writing anything")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root or os.environ.get("PROJECT_ROOT")
    if not root:
        sys.exit("Provide --root or set $PROJECT_ROOT")

    _inject_ga_dummies()

    pattern = os.path.join(root, "**", "pickled_ga_object.pkl")
    pkl_files = sorted(glob.glob(pattern, recursive=True))

    if not pkl_files:
        sys.exit(f"No pickled_ga_object.pkl files found under {root}")

    print(f"Found {len(pkl_files)} pkl file(s)"
          + (" — dry run" if args.dry_run else ""))

    total_pkl_kb = total_json_kb = 0.0
    for pkl_path in pkl_files:
        print(pkl_path)
        pkl_kb, json_kb = extract(pkl_path, dry_run=args.dry_run)
        if pkl_kb is not None:
            total_pkl_kb += pkl_kb
            total_json_kb += json_kb

    print(f"\nTotal: {total_pkl_kb / 1024:.1f} MB pkl  ->  "
          f"{total_json_kb / 1024:.1f} MB json  "
          f"({100 * total_json_kb / total_pkl_kb:.0f}% of original)")

    if not args.dry_run:
        print("\nAdd to .gitignore:")
        print("  pickled_ga_object.pkl")


if __name__ == "__main__":
    main()
