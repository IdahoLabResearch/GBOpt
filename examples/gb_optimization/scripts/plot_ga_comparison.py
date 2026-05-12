#!/usr/bin/env python3
"""
plot_ga_comparison_panels.py
----------------------------
Two-panel publication-ready GA convergence comparison for a single boundary.
Left panel: Ni data. Right panel: Fe data.
Each panel shows all runs (standard + high-energy) as best-so-far trajectories
plotted as 1 + (E_GB - E_GB^lit) vs generation, with a per-panel inset showing
the early convergence region.

Usage:
    python scripts/plot_ga_comparison_panels.py --boundary sigma5_310_STGB
    python scripts/plot_ga_comparison_panels.py --boundary sigma5_310_STGB \\
        --materials Ni Fe
    python scripts/plot_ga_comparison_panels.py --boundary sigma5_310_STGB \\
        --inset-y-hi 1.25
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import types
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from cmcrameri import cm as cmc
    _HAS_CMCRAMERI = True
except ImportError:
    import matplotlib.cm as cmc
    _HAS_CMCRAMERI = False
    warnings.warn("cmcrameri not found; using matplotlib default colormaps.")

from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FixedLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
# Within each panel: standard=_pal[0], high_energy=_pal[2]
# ---------------------------------------------------------------------------
if _HAS_CMCRAMERI:
    _pal = list(cmc.batlowS.colors)
else:
    _pal = [plt.cm.tab10(i) for i in range(10)]

PANEL_COLORS = {
    "standard":    _pal[0],
    "high_energy": _pal[2],
}
LINESTYLES = {"standard": "-",        "high_energy": "--"}
INIT_LABELS = {"standard": "Standard", "high_energy": "High-energy"}

# Fixed x window for Fe inset (generations 0-50)
FE_INSET_X_END = 50


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


def _bold_4f(x, pos):
    if USE_LATEX:
        return r"$\mathbf{" + f"{x:.4f}" + r"}$"
    return f"{x:.4f}"


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

def _load_json(fpath):
    try:
        with open(fpath) as fh:
            data = json.load(fh)
        obj = types.SimpleNamespace(
            population_size=data["population_size"],
            generations=data["n_generations"],
            GBE_vals=data["gbe_vals"],
        )
        obj.GBE_vals = [
            [e for e in g if e < PENALTY_THRESHOLD]
            for g in obj.GBE_vals
        ]
        obj.GBE_vals = [g for g in obj.GBE_vals if g]
        return obj
    except Exception as exc:
        warnings.warn(f"Could not load {fpath}: {exc}")
        return None


def load_runs(material, boundary, root):
    base_ga = os.path.join(root, material, boundary, "GA")
    patterns = {
        "standard":    os.path.join(base_ga, "standard",    "run*", "ga_data.json"),
        "high_energy": os.path.join(base_ga, "high_energy", "run*", "ga_data.json"),
    }
    result = {}
    for group, pattern in patterns.items():
        objs = [_load_json(p) for p in sorted(glob.glob(pattern))]
        objs = [o for o in objs if o is not None]
        if objs:
            result[group] = objs
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
        sys.exit(
            f"No literature GBE found for boundary='{boundary}', "
            f"material='{material}' in literature_gbe.toml"
        )


# ---------------------------------------------------------------------------
# Energy processing
# ---------------------------------------------------------------------------

def min_per_gen(obj):
    return np.array([min(g) for g in obj.GBE_vals if g])


def best_so_far(vals):
    return np.minimum.accumulate(vals)


def transform(vals, gamma_lit):
    """1 + (E_GB - E_lit) so literature value maps to 1.0."""
    return 1.0 + (vals - gamma_lit)


def _count_converged_ga(objs, gamma_lit, tol=0.005):
    """Count GA runs whose best-so-far final value is within tol of 1.0."""
    count = 0
    for obj in objs:
        vals = min_per_gen(obj)
        bsf = best_so_far(vals)
        tr = transform(bsf, gamma_lit)
        if float(tr[-1]) <= 1.0 + tol:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Inset window detection
# ---------------------------------------------------------------------------

def _find_convergence_window(all_tr, x_end_override=None, y_hi_override=None):
    """
    Find inset window parameters.

    x-range: generation 0 to when the last run settles, or x_end_override.
    y-range: 1.0 (bottom) to 90th percentile of data in window (top),
             capped at 1.5. Override with y_hi_override for a fixed bound.

    Returns (x_end, y_lo, y_hi).
    """
    max_gens = max(len(t) for t in all_tr)
    padded = np.array([
        np.pad(t, (0, max_gens - len(t)), mode="edge") for t in all_tr
    ])

    if x_end_override is not None:
        x_end = min(x_end_override, max_gens - 1)
    else:
        median_final = float(np.median([t[-1] for t in all_tr]))
        settle_threshold = median_final * 1.001
        settle_gens = []
        for run in padded:
            hits = np.where(run <= settle_threshold)[0]
            settle_gens.append(int(hits[0]) if len(hits) > 0 else max_gens - 1)
        x_end = min(int(np.max(settle_gens)) + 5, max_gens - 1)

    window_data = padded[:, :x_end + 1].flatten()
    y_lo = 1.0

    if y_hi_override is not None:
        y_hi = y_hi_override
    else:
        y_hi = min(float(np.percentile(window_data, 90)), 1.5)
        y_hi = max(y_hi, 1.01)

    return x_end, y_lo, y_hi


# ---------------------------------------------------------------------------
# Panel plotting
# ---------------------------------------------------------------------------

def _plot_panel(ax, runs, gamma_lit, material, all_plot_data):
    """
    Plot all runs for a single material on ax.
    Populates all_plot_data in-place with (gens, tr, color, lw, ls, alpha).
    """
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0,
               alpha=0.8, zorder=1)

    for group, objs in runs.items():
        color = PANEL_COLORS.get(group, _pal[3])
        ls = LINESTYLES[group]
        n_conv = _count_converged_ga(objs, gamma_lit)
        n_total = len(objs)
        label = _tex(rf"\textbf{{{INIT_LABELS[group]}}} ({n_conv}/{n_total})")
        for run_idx, obj in enumerate(objs):
            vals = min_per_gen(obj)
            bsf = best_so_far(vals)
            tr = transform(bsf, gamma_lit)
            gens = np.arange(len(tr))
            lbl = label if run_idx == 0 else None
            n = len(objs)
            lw = 1.5 if n == 1 else (1.8 if run_idx == 0 else 0.8)
            alpha = 1.0 if n == 1 else (1.0 if run_idx == 0 else 0.45)
            ax.plot(gens, tr, color=color, linewidth=lw,
                    linestyle=ls, alpha=alpha, label=lbl)
            all_plot_data.append((gens, tr, color, lw, ls, alpha))

    ax.set_title(_tex(rf"\textbf{{{material}}}"), fontsize=16, pad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_bold_int))
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_2f))
    _apply_style(ax)


def _add_inset(ax, all_plot_data, all_tr,
               x_end_override=None, y_hi_override=None):
    """
    Add convergence inset to ax. Leader line connects from (x=0, y=1.0)
    on main axis to the bottom-left of the inset.
    """
    x_end, y_lo, y_hi = _find_convergence_window(
        all_tr,
        x_end_override=x_end_override,
        y_hi_override=y_hi_override,
    )

    axins = inset_axes(
        ax, width="100%", height="100%",
        bbox_to_anchor=(0.45, 0.18, 0.50, 0.45),
        bbox_transform=ax.transAxes,
        loc="lower left", borderpad=0,
    )
    axins.set_facecolor("white")
    axins.set_zorder(10)
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")
    axins.tick_params(labelsize=11, width=1.0, length=4)

    axins.axhline(1.0, color="gray", linestyle=":", linewidth=1.0,
                  alpha=0.8, zorder=1)

    for gens, tr, color, lw, ls, alpha in all_plot_data:
        mask = gens <= x_end
        if mask.any():
            axins.plot(gens[mask], tr[mask], color=color,
                       linewidth=lw, linestyle=ls, alpha=alpha)

    axins.set_xlim(0, x_end)
    axins.set_ylim(y_lo, y_hi)
    axins.xaxis.set_major_formatter(FuncFormatter(_bold_int))
    axins.yaxis.set_major_formatter(FuncFormatter(_bold_4f))

    mid_hi = round((1.0 + y_hi) / 2, 4)
    ticks = np.array([1.0, mid_hi])
    axins.yaxis.set_major_locator(FixedLocator(ticks))
    axins.yaxis.set_minor_locator(plt.NullLocator())

    con = ConnectionPatch(
        xyA=(0, 1.0), coordsA=ax.transData,
        xyB=(0, 0),   coordsB=axins.transAxes,
        color="gray", linewidth=0.8, linestyle="-", alpha=0.5,
    )
    ax.add_artist(con)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-panel GA convergence comparison (one panel per material)")
    p.add_argument("--boundary",  default="sigma5_310_STGB")
    p.add_argument("--materials", nargs="+", default=["Ni", "Fe"])
    p.add_argument(
        "--inset-y-hi", type=float, default=None,
        help=(
            "Fixed upper y-bound for insets (e.g. 1.25). "
            "If omitted, 90th percentile of window data is used (capped at 1.5)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.parent

    material_data = {}
    for mat in args.materials:
        gamma_lit = load_literature_gbe(root, args.boundary, mat)
        runs = load_runs(mat, args.boundary, root)
        if not runs:
            warnings.warn(f"No GA runs found for {mat}/{args.boundary} — skipping.")
            continue
        material_data[mat] = {"gamma_lit": gamma_lit, "runs": runs}
        for group, objs in runs.items():
            print(f"  {mat} {group}: {len(objs)} run(s)")

    if not material_data:
        sys.exit("No data found.")

    n_panels = len(material_data)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, mat in zip(axes, material_data):
        mdata = material_data[mat]
        gamma_lit = mdata["gamma_lit"]
        runs = mdata["runs"]

        all_tr = []
        for group, objs in runs.items():
            for obj in objs:
                vals = min_per_gen(obj)
                bsf = best_so_far(vals)
                tr = transform(bsf, gamma_lit)
                all_tr.append(tr)

        all_plot_data = []
        _plot_panel(ax, runs, gamma_lit, mat, all_plot_data)

        ax.set_xlabel(_tex(r"\textbf{Generation}"), fontsize=14)
        ax.legend(fontsize=13, frameon=True, framealpha=0.95,
                  loc="upper right", edgecolor="black",
                  fancybox=False, borderpad=0.5)

        x_end_override = FE_INSET_X_END if mat == "Fe" else None
        _add_inset(ax, all_plot_data, all_tr,
                   x_end_override=x_end_override,
                   y_hi_override=args.inset_y_hi)

    axes[0].set_ylabel(
        _tex(
            r"$\mathbf{1 + (E_\mathrm{GB} - E_\mathrm{GB}^\mathrm{lit})}$"
            r" \textbf{(J/m$\mathbf{^2}$)}"
        ),
        fontsize=13,
    )

    fig.tight_layout()

    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    stem = f"ga_convergence_panels_{'_'.join(material_data.keys())}_{args.boundary}"
    for ext in ("png", "pdf"):
        out = os.path.join(fig_dir, f"{stem}.{ext}")
        plt.savefig(out, dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
