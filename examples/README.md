# GBOpt Examples

This directory contains worked examples accompanying the manuscript:

> **GBOpt: A Python Package for Grain Boundary Structure Optimization**
> J.C. French, C.V. Bhave (submitted)

Each subdirectory is a self-contained example with its own README, input
files, committed results data, and plotting scripts. The examples are
designed to be used in one of two ways:

- **Reproduce the manuscript figures** — all optimization results are
  committed to the repository. Only Python dependencies are required;
  no HPC system or LAMMPS installation is needed.

- **Re-run the optimization** — the full optimization requires LAMMPS
  and a SLURM-based HPC cluster. See the README in each subdirectory
  for details.

## Examples

| Directory | Description |
|-----------|-------------|
| [`gb_optimization/`](gb_optimization/) | GA and MC optimization of grain boundary energy across six boundaries and three materials (Fe, Ni, Si) |
| [`density_optimization/`](density_optimization/) | GA optimization of GB core atom density for the Σ5[001]{310} boundary in Fe and Ni |

## Dependencies

GBOpt and its required dependencies (NumPy, SciPy, pandas, matplotlib,
numba, spglib) must be installed:

```bash
pip install -e /path/to/GBOpt
```

The plotting scripts additionally use
[cmcrameri](https://github.com/callumrollo/cmcrameri) for perceptually
uniform colormaps:

```bash
pip install cmcrameri
```

If `cmcrameri` is not available the scripts will fall back to matplotlib
default colormaps. TOML config files are read with the standard library
`tomllib` module (Python ≥ 3.11) or the
[tomli](https://github.com/hukkin/tomli) backport for earlier versions:

```bash
pip install tomli   # only needed for Python < 3.11
```
