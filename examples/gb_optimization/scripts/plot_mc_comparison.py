#!/usr/bin/env python3
"""
plot_mc_comparison_panels.py
----------------------------
Publication-ready MC GBE vs step figure for a single material and boundary.
Shows all runs as a band (IQR shaded) with the best run highlighted.
Y-axis is normalized so the literature GBE maps to 1.0.
Convergence count is shown in each legend entry.

Run once per material:
    python scripts/plot_mc_comparison_panels.py --material Ni --boundary sigma5_310_STGB
    python scripts/plot_mc_comparison_panels.py --material Fe --boundary sigma5_310_STGB
    python scripts/plot_mc_comparison_panels.py --material Fe --boundary sigma5_310_STGB \\
        --e-accept 0.01
"""

import argparse
import glob
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from cmcrameri import cm as cmc
    _HAS_CMCRAMERI = True
except ImportError:
    import matplotlib.cm as cmc
    _HAS_CMCRAMERI = False
    warnings.warn("cmcrameri not found; using matplotlib default colormaps.")

from matplotlib.ticker import FuncFormatter

try:
    import tomllib
except ImportError:
    import tomli as tomllib

PENALTY_THRESHOLD = 1.0e29

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------


def _configure_mpl_style() -> bool:
    """
    Apply publication rcParams. Returns True if LaTeX is available.
    """
    latex_path = "/apps/local/latex/bin/x86_64-linux"
    if latex_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = latex_path + os.pathsep + os.environ.get("PATH", "")

    use_latex = shutil.which("pdflatex") is not None

    base = {
        "font.size":        18,
        "axes.labelsize":   18,
        "axes.titlesize":   20,
        "xtick.labelsize":  16,
        "ytick.labelsize":  16,
        "legend.fontsize":  14,
        "axes.labelweight": "bold",
        "lines.linewidth":  2,
        "lines.markersize": 6,
        "figure.dpi":       300,
    }

    if use_latex:
        base.update({
            "font.family":   "serif",
            "text.usetex":   True,
            "pgf.texsystem": "pdflatex",
        })
    else:
        base.update({
            "font.family": "DejaVu Sans",
            "text.usetex": False,
        })

    plt.rcParams.update(base)
    return use_latex


USE_LATEX = _configure_mpl_style()

# ---------------------------------------------------------------------------
# Colors: batlowS categorical palette, skipping index 1 (light pink)
# standard=_pal[0], high_energy=_pal[2] — consistent with GA panels
# ---------------------------------------------------------------------------
if _HAS_CMCRAMERI:
    _pal = list(cmc.batlowS.colors)
else:
    _pal = [plt.cm.tab10(i) for i in range(10)]

GROUP_COLORS = {
    "standard":    _pal[0],
    "high_energy": _pal[2],
}
GROUP_LABELS = {
    "standard":    "Standard initialization",
    "high_energy": "High-energy restart",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tex(s: str) -> str:
    """Pass through LaTeX markup if available, else strip it."""
    if USE_LATEX:
        return s
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$\\mathbf\{([^}]*)\}\$", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    return s


def _bold_int(x, pos):
    if USE_LATEX:
        return r"$\mathbf{" + f"{int(x)}" + r"}$"
    return f"{int(x)}"


def _bold_2f(x, pos):
    if USE_LATEX:
        return r"$\mathbf{" + f"{x:.2f}" + r"}$"
    return f"{x:.2f}"


def _apply_style(ax):
    ax.tick_params(labelsize=16, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_mc_file(fpath):
    run_num = re.sub(r"^run", "", os.path.basename(os.path.dirname(fpath)))
    df = pd.read_csv(fpath, sep=r"\s+", header=None,
                     names=["label", "gbe", "accepted"])
    df["accepted"] = df["accepted"].apply(lambda x: str(x).strip() == "True")
    df["gbe"] = pd.to_numeric(df["gbe"], errors="coerce")
    df = df.dropna(subset=["gbe"])
    df = df[df["gbe"] < PENALTY_THRESHOLD].reset_index(drop=True)
    return {"run": run_num, "df": df}


def load_mc_runs(material, boundary, root):
    base_mc = os.path.join(root, material, boundary, "MC")
    result = {}

    pattern = os.path.join(base_mc, "standard", "run*", "gbe_vals.txt")
    runs = []
    for fpath in sorted(glob.glob(pattern)):
        try:
            runs.append(_load_mc_file(fpath))
        except Exception as exc:
            warnings.warn(f"Could not load {fpath}: {exc}")
    if runs:
        result["standard"] = runs

    he_pattern = os.path.join(base_mc, "high_energy", "run*", "gbe_vals.txt")
    runs = []
    for fpath in sorted(glob.glob(he_pattern)):
        try:
            runs.append(_load_mc_file(fpath))
        except Exception as exc:
            warnings.warn(f"Could not load {fpath}: {exc}")
    if runs:
        result["high_energy"] = runs

    return result


def load_literature_gbe(root, boundary, material):
    cfg_path = os.path.join(root, "config", "literature_gbe.toml")
    if not os.path.exists(cfg_path):
        sys.exit(f"literature_gbe.toml not found at {cfg_path}")
    with open(cfg_path, "rb") as fh:
        cfg = tomllib.load(fh)
    try:
        return float(cfg[boundary][material])
    except KeyError:
        warnings.warn(
            f"No literature GBE for boundary='{boundary}', material='{material}'")
        return None


# ---------------------------------------------------------------------------
# Energy processing
# ---------------------------------------------------------------------------

def _transform(gbe, gamma_lit):
    return 1.0 + (np.asarray(gbe, dtype=float) - gamma_lit)


def _best_run_idx(runs):
    return int(np.argmin([r["df"]["gbe"].min() for r in runs]))


def _count_converged(runs, gamma_lit, tol=0.005):
    """Count runs whose minimum transformed GBE is within tol of 1.0."""
    count = 0
    for run in runs:
        if float(_transform(run["df"]["gbe"].min(), gamma_lit)) <= 1.0 + tol:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_mc_panel(ax, mc_runs, gamma_lit, material):
    """
    Band plot of transformed GBE vs MC step for one material.
    IQR band + best run highlighted per group.
    Circle marker at first group minimum on best run.
    Convergence count embedded in legend label.
    """
    ax.axhline(1.0, color="gray", linestyle=":",
               linewidth=1.0, alpha=0.8, zorder=1)

    for group, runs in mc_runs.items():
        color = GROUP_COLORS.get(group, _pal[3])
        base_lbl = GROUP_LABELS.get(group, group)
        best_idx = _best_run_idx(runs)
        n_conv = _count_converged(runs, gamma_lit)
        n_total = len(runs)
        label = _tex(rf"\textbf{{{base_lbl}}} ({n_conv}/{n_total})")

        max_len = max(len(r["df"]) for r in runs)
        steps = np.arange(max_len)
        tr_array = np.full((len(runs), max_len), np.nan)
        for i, run in enumerate(runs):
            tr = _transform(run["df"]["gbe"].values, gamma_lit)
            tr_array[i, :len(tr)] = tr

        n_active = (~np.isnan(tr_array)).sum(axis=0)
        valid = n_active >= 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            env_min = np.nanmin(tr_array, axis=0)
            env_max = np.nanmax(tr_array, axis=0)
            p25 = np.nanpercentile(tr_array, 25, axis=0)
            p75 = np.nanpercentile(tr_array, 75, axis=0)

        ax.fill_between(steps[valid], env_min[valid], env_max[valid],
                        color=color, alpha=0.18, linewidth=0)
        ax.fill_between(steps[valid], p25[valid], p75[valid],
                        color=color, alpha=0.32, linewidth=0)

        best_tr = _transform(runs[best_idx]["df"]["gbe"].values, gamma_lit)
        ax.plot(np.arange(len(best_tr)), best_tr,
                color=color, linewidth=2.2, alpha=1.0, label=label)

        group_min_tr = float(np.nanmin(tr_array))
        hits = np.where(best_tr <= group_min_tr * 1.0001)[0]
        if len(hits) > 0:
            step = int(hits[0])
            ax.scatter([step], [best_tr[step]], color=color,
                       marker="o", s=60, zorder=6,
                       facecolors="none", edgecolors=color, linewidths=1.8)

    ax.set_title(_tex(rf"\textbf{{{material}}}"), fontsize=18, pad=6)
    ax.set_xlabel(_tex(r"\textbf{Monte Carlo Step}"), fontsize=14)
    ax.set_ylabel(
        _tex(
            r"$\mathbf{1 + (E_\mathrm{GB} - E_\mathrm{GB}^\mathrm{lit})}$"
            r" \textbf{(J/m$\mathbf{^2}$)}"
        ),
        fontsize=13,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_bold_int))
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_2f))
    ax.legend(fontsize=13, frameon=True, framealpha=0.95,
              loc="upper right", edgecolor="black",
              fancybox=False, borderpad=0.5)
    _apply_style(ax)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-material MC GBE vs step band plot")
    p.add_argument("--material",  required=True)
    p.add_argument("--boundary",  required=True)
    p.add_argument("--e-accept",  default="0.1",
                   help="MC acceptance energy threshold (default: 0.1)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

    mc_runs = load_mc_runs(args.material, args.boundary, root)
    if not mc_runs:
        sys.exit(f"No MC runs found for {args.material}/{args.boundary}.")
    for group, runs in mc_runs.items():
        print(f"  {args.material} {group}: {len(runs)} run(s)")

    gamma_lit = load_literature_gbe(root, args.boundary, args.material)
    if gamma_lit is None:
        sys.exit("Literature GBE not found — cannot transform y-axis.")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_mc_panel(ax, mc_runs, gamma_lit, args.material)
    fig.tight_layout()

    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    stem = (f"mc_gbe_{args.material}_{args.boundary}"
            f"_e{args.e_accept.replace('.', 'p')}")
    for ext in ("png", "pdf"):
        out = os.path.join(fig_dir, f"{stem}.{ext}")
        plt.savefig(out, dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
