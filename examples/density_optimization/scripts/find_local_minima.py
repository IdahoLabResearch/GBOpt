#!/usr/bin/env python
"""
find_local_minima.py
Print the generation, candidate dump path, [n], and GBE for each local minimum
on the lower envelope, plus the global minimum.

Usage:
    python scripts/find_local_minima.py --material Fe
"""

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# --- copy these from plot_density_progress.py ---
N_BULK_PLANE = 45
BOUNDARY = "sigma5_310_STGB"


def parse_atom_count(path):
    with open(path) as fh:
        for line in fh:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                return int(next(fh).strip())
    raise ValueError(f"No atom count in {path}")


def _process_candidate(args):
    gen_idx, dump_path, gbe_path = args
    try:
        n_total = parse_atom_count(dump_path)
        n_val = (n_total % N_BULK_PLANE) / N_BULK_PLANE
        gbe = float(Path(gbe_path).read_text().split()[1])
        return (gen_idx, dump_path, n_val, gbe)
    except Exception as exc:
        print(f"  Skipping {Path(dump_path).name}: {exc}", file=sys.stderr)
        return None


def lower_envelope_binned(xs, ys, n_bins=60):
    xs, ys = np.array(xs), np.array(ys)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, mins = [], []
    for i in range(n_bins):
        mask = (xs >= bin_edges[i]) & (xs < bin_edges[i + 1])
        if mask.sum() > 0:
            centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            mins.append(ys[mask].min())
    return np.array(centers), np.array(mins)


# -----------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--material", required=True)
parser.add_argument(
    "--prominence",
    type=float,
    default=0.02,
    help="Peak prominence threshold for local minima detection",
)
parser.add_argument("--hull-max", type=float, default=2.5)
args = parser.parse_args()

project_root = Path(__file__).parent.parent
workdir = (
    project_root
    / args.material
    / BOUNDARY
    / "GA"
    / "standard"
    / "workdir.1"
)

# Collect work items
work_items = []
for gen_dir in sorted(workdir.glob("gen_*"), key=lambda p: int(p.name.split("_")[1])):
    gen_idx = int(gen_dir.name.split("_")[1])
    for dump_path in sorted(gen_dir.glob("final_GA_1_g*_c*.dump")):
        m = re.search(r"_c(\d+)\.dump$", dump_path.name)
        if not m:
            continue
        gbe_path = gen_dir / f"temp_GA_1_g{gen_idx}_c{m.group(1)}.dump_GBE.txt"
        if gbe_path.exists():
            work_items.append((gen_idx, str(dump_path), str(gbe_path)))

print(f"Reading {len(work_items)} dump files...")
records = []
with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    for r in pool.map(_process_candidate, work_items, chunksize=50):
        if r:
            records.append(r)

all_n = np.array([r[2] for r in records])
all_gbe = np.array([r[3] for r in records])

# Lower envelope + local minima (same logic as plot script)
mask = all_gbe <= args.hull_max
lx, ly = lower_envelope_binned(all_n[mask], all_gbe[mask], n_bins=60)
ly_smooth = uniform_filter1d(ly, size=5, mode="nearest")
peak_idx, _ = find_peaks(-ly_smooth, prominence=args.prominence)

# Refine to nearest raw minimum within ±3 bins
refined = []
for i in peak_idx:
    lo, hi = max(0, i - 3), min(len(ly), i + 4)
    refined.append(lo + np.argmin(ly[lo:hi]))
min_idx = np.unique(refined)

# Always include global minimum
global_min = np.argmin(ly)
all_special = np.unique(np.append(min_idx, global_min))

print(
    f"\nFound {len(min_idx)} local minima + global minimum ({len(all_special)} unique):\n"
)
print(f"{'[n]':>8}  {'GBE (J/m²)':>12}  {'Gen':>6}  {'Dump file'}")
print("-" * 80)

for idx in sorted(all_special, key=lambda i: lx[i]):
    n_target = lx[idx]
    gbe_target = ly[idx]
    # Find the actual candidate closest to this envelope point
    dist = np.abs(all_n - n_target)
    # Among candidates within a small [n] window, pick lowest GBE
    window_mask = dist < (0.5 / 60)  # half a bin width
    if not window_mask.any():
        window_mask = dist == dist.min()
    candidates_in_window = [
        (records[i][0], records[i][1], records[i][2], records[i][3])
        for i in np.where(window_mask)[0]
    ]
    best = min(candidates_in_window, key=lambda x: x[3])
    gen, dump, n_val, gbe = best
    tag = " <-- global min" if idx == global_min else ""
    print(f"{n_val:8.4f}  {gbe:12.6f}  {gen:6d}  {Path(dump).name}{tag}")
