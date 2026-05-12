#!/usr/bin/env python3
"""
plot_density_progress.py
------------------------
Plot density-optimization progress using the Zhu et al. (2018) [n] metric:

    [n] = (n_gb_core / n_perfect) mod 1

where n_gb_core is the number of atoms in the GB core slab and n_perfect = 45
is the number of primitive 2D unit cells in the GB plane (5×9 supercell).
These values are stored directly in density_gbe_log.json as the `density`
field (= n_gb_core / n_perfect); [n] is recovered at plot time via % 1.0.

Reference: Zhu et al., Nature Communications 9, 467 (2018).

Usage:
    python scripts/plot_density_progress.py --material Ni
    python scripts/plot_density_progress.py --material Fe

Output:
    {data_dir}/{material}/sigma5_310_STGB/figures/density_progress.{png,pdf}
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from cmcrameri import cm as cmc
    _HAS_CMCRAMERI = True
except ImportError:
    import matplotlib.cm as cmc
    _HAS_CMCRAMERI = False
    import warnings
    warnings.warn("cmcrameri not found; using matplotlib default colormaps.")

from matplotlib.ticker import FuncFormatter
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOUNDARY = "sigma5_310_STGB"

# GBE above this is excluded from the lower envelope (but still plotted)
GBE_HULL_MAX = {
    "Ni": 1.8,
    "Fe": 2.5,
}
GBE_HULL_MAX_DEFAULT = 1.8

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
        "legend.fontsize":  15,
        "axes.labelweight": "bold",
        "lines.linewidth":  2,
        "lines.markersize": 6,
        "figure.dpi":       600,
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
# Helpers
# ---------------------------------------------------------------------------

def _tex(s: str) -> str:
    """Pass through LaTeX markup if available, else strip it."""
    if USE_LATEX:
        return s
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$\\mathbf\{([^}]*)\}\$", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    return s


def lower_envelope_binned(xs, ys, n_bins=60):
    xs, ys = np.array(xs), np.array(ys)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_mins = [], []
    for i in range(n_bins):
        mask = (xs >= bin_edges[i]) & (xs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_mins.append(ys[mask].min())
    return np.array(bin_centers), np.array(bin_mins)


# ---------------------------------------------------------------------------
# Tick formatters
# ---------------------------------------------------------------------------

def _bold_int(x, pos):
    if USE_LATEX:
        return r"$\mathbf{" + f"{int(x)}" + r"}$"
    return f"{int(x)}"


def _bold_float2(x, pos):
    if USE_LATEX:
        return r"$\mathbf{" + f"{x:.2f}" + r"}$"
    return f"{x:.2f}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_log(density_log: dict) -> dict[int, list[tuple]]:
    """
    Build per-generation records from density_gbe_log.json.

    The JSON `density` field stores n_gb_core / n_perfect per candidate.
    [n] = density % 1.0, giving the Zhu et al. fractional GB atomic density
    in [0, 1).
    """
    gen_records: dict[int, list[tuple]] = {}
    for gen_idx, (density_gen, gbe_gen) in enumerate(
        zip(density_log["density"], density_log["gbe_vals"])
    ):
        gen_records[gen_idx] = [
            (d % 1.0, gbe)
            for d, gbe in zip(density_gen, gbe_gen)
        ]
    return gen_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot density optimization progress")
    parser.add_argument("--material", required=True, help="Ni or Fe")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Directory layout — adjust script_dir if your repository structure
    # differs from:
    #   <repo_root>/{material}/sigma5_310_STGB/density_gbe_log.json
    # -----------------------------------------------------------------------
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / args.material / BOUNDARY
    figures_dir = data_dir / "figures"

    log_path = data_dir / "density_gbe_log.json"
    if not log_path.exists():
        print(f"density_gbe_log.json not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with open(log_path) as fh:
        density_log = json.load(fh)

    gen_records = load_from_log(density_log)
    if not gen_records:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    gens_sorted = sorted(gen_records.keys())
    hull_max = GBE_HULL_MAX.get(args.material, GBE_HULL_MAX_DEFAULT)
    title_prefix = rf"\textbf{{{args.material}}} --- "
    x_label = r"\textbf{$[n]$ (fraction of (310) plane)}"

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    sm = mpl.cm.ScalarMappable(
        cmap=cmc.batlow if _HAS_CMCRAMERI else mpl.cm.viridis,
        norm=mpl.colors.Normalize(vmin=gens_sorted[0], vmax=gens_sorted[-1]),
    )
    sm.set_array([])

    if _HAS_CMCRAMERI:
        pal = list(cmc.batlowS.colors)
    else:
        pal = [plt.cm.tab10(i) for i in range(10)]
    c_mean = pal[4]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: mean [n] and IQR per generation ────────────────────────
    ax = axes[0]

    mean_n = [np.mean([e[0] for e in gen_records[g]]) for g in gens_sorted]
    p25 = [np.percentile([e[0] for e in gen_records[g]], 25) for g in gens_sorted]
    p75 = [np.percentile([e[0] for e in gen_records[g]], 75) for g in gens_sorted]

    ax.fill_between(
        gens_sorted, p25, p75, alpha=0.25, color=c_mean,
        label=_tex(r"\textbf{IQR}"),
    )
    ax.plot(
        gens_sorted, mean_n,
        color=c_mean, lw=1.8, linestyle="--",
        label=_tex(r"\textbf{Mean}"),
    )

    ax.set_xlabel(_tex(r"\textbf{Generation}"), fontsize=20)
    ax.set_ylabel(_tex(x_label), fontsize=18)
    ax.set_title(_tex(title_prefix + r"$[n]$ vs.\ generation"), fontsize=22, pad=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=15, framealpha=0.95, frameon=True,
              edgecolor="black", fancybox=False)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
    ax.tick_params(labelsize=18, width=1.5, length=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_bold_int))
    ax.yaxis.set_major_formatter(FuncFormatter(_bold_float2))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # ── Right panel: GBE vs [n] scatter, coloured by generation ───────────
    ax2 = axes[1]

    all_n, all_gbe = [], []
    for g in gens_sorted:
        xs = [e[0] for e in gen_records[g]]
        ys = [e[1] for e in gen_records[g]]
        ax2.scatter(
            xs, ys,
            c=[sm.to_rgba(g)] * len(xs),
            s=18, alpha=0.7, linewidths=0, zorder=2,
        )
        all_n.extend(xs)
        all_gbe.extend(ys)

    # Lower envelope
    hull_n = [x for x, y in zip(all_n, all_gbe) if y <= hull_max]
    hull_gbe = [y for x, y in zip(all_n, all_gbe) if y <= hull_max]

    if len(hull_n) > 5:
        lx, ly = lower_envelope_binned(hull_n, hull_gbe, n_bins=60)
        ly_smooth = uniform_filter1d(ly, size=5, mode="nearest")
        min_idx_approx, _ = find_peaks(-ly_smooth, prominence=0.02)

        window = 3
        refined = []
        for i in min_idx_approx:
            lo, hi = max(0, i - window), min(len(ly), i + window + 1)
            refined.append(lo + np.argmin(ly[lo:hi]))
        min_idx = np.unique(refined)

        global_min_idx = np.argmin(ly)
        metastable_idx = min_idx[min_idx > global_min_idx]

        ax2.plot(
            lx, ly, color="black", lw=1.5, linestyle="--", zorder=3,
            label=_tex(r"\textbf{Lower envelope}"),
        )
        ax2.plot(
            lx, ly_smooth, color="black", lw=1.2, linestyle="-",
            alpha=0.5, zorder=3,
        )

        if len(min_idx) > 0:
            ax2.scatter(
                lx[min_idx], ly[min_idx],
                marker="v", s=70, color="black", zorder=5,
                label=_tex(r"\textbf{Local min}"),
            )

        if len(metastable_idx) > 0:
            mx = metastable_idx[-1]
            ax2.axvline(lx[mx], color="crimson", lw=1.2, linestyle=":", zorder=4)
            ax2.scatter(
                [lx[mx]], [ly[mx]],
                marker="*", s=180, color="crimson", zorder=6,
                label=_tex(rf"\textbf{{Max metastable: $[n]$={lx[mx]:.3f}}}"),
            )

    ax2.legend(
        fontsize=13, framealpha=0.95, frameon=True,
        edgecolor="black", fancybox=False, loc="upper right",
    )

    cbar = fig.colorbar(sm, ax=ax2, pad=0.02)
    cbar.set_label(_tex(r"\textbf{Generation}"), fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    ax2.set_xlabel(_tex(x_label), fontsize=18)
    ax2.set_ylabel(_tex(r"\textbf{GB Energy (J/m$^\mathbf{2}$)}"), fontsize=20)
    ax2.set_title(_tex(title_prefix + r"energy vs.\ $[n]$"), fontsize=22, pad=10)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
    ax2.tick_params(labelsize=18, width=1.5, length=6)
    ax2.xaxis.set_major_formatter(FuncFormatter(_bold_float2))
    ax2.yaxis.set_major_formatter(FuncFormatter(_bold_float2))
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = figures_dir / f"density_progress.{ext}"
        plt.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
