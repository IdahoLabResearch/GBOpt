# Density Optimization Example

This example demonstrates the use of GBOpt to maximize the density of
grain boundary (GB) core atoms for the Σ5[001]{310} boundary in Fe and Ni.
The optimization uses the genetic algorithm (GA) with a custom objective
function based on the GB core atom count metric of Zhu *et al.* (2018).

**Reference:** Q. Zhu *et al.*, Nature Communications **9**, 467 (2018).

## Directory structure

```
density_optimization/
├── lammps/
│   ├── lmp.in                 # LAMMPS input script
│   ├── min_strat.commands     # minimization strategy commands
│   ├── Fe/potential.setup
│   └── Ni/potential.setup
├── scripts/
│   ├── optimize.py            # density optimization (GA)
│   ├── plot_density.py        # plot optimization progress
│   ├── find_local_minima.py   # identify local minima in the log
│   └── slurm_utils.py         # SLURM job submission utilities
├── Fe/
│   └── sigma5_310_STGB/
│       ├── best_density_structure.txt  # highest-density structure found
│       └── density_gbe_log.json         # log of (density, GBE) per evaluation
└── Ni/
    └── sigma5_310_STGB/
        ├── best_density_structure.txt
        └── density_gbe_log.json
```

## The density metric

The optimization maximizes the normalized GB core atom density [*n*],
defined following Zhu *et al.* (2018) as:

```
[n] = (N_total mod N_bulk_plane) / N_bulk_plane
```

where *N*_total is the total atom count of the simulation cell and
*N*_bulk\_plane is the number of atoms in one bulk plane parallel to the
GB (45 atoms for both Fe BCC and Ni FCC in the 5×9 (310) supercell used
here).

## Track 1 — Reproduce manuscript figures

The optimization results are included in the repository. No LAMMPS
installation or HPC system is required.

### Setup

```bash
pip install -e /path/to/GBOpt
pip install cmcrameri          # optional but recommended
pip install tomli              # only needed for Python < 3.11
```

### Generate plots

Plots are saved to `{material}/sigma5_310_STGB/figures/`:

```bash
python scripts/plot_density.py --material Fe
python scripts/plot_density.py --material Ni
```

Each call produces a two-panel figure:
- **Left panel:** mean [*n*] ± IQR across the population per generation
- **Right panel:** GB energy vs [*n*] scatter coloured by generation,
  with lower envelope and identified local minima

> **Note:** When `workdir.1/` is not present (as when cloning from the
> repository), the script falls back to plotting from
> `density_gbe_log.json` alone. Generation indices are approximated by
> dividing evaluations into blocks of 50 (the population size used in
> the manuscript). The scatter plot will be identical; the left panel
> will show approximate generation numbers.

## Track 2 — Re-run the optimization

### Requirements

- LAMMPS compiled with the KOKKOS and MANYBODY packages. The results in
  the manuscript were produced with LAMMPS version `patch_30Mar2026`
  on INL's Teton cluster.
- A SLURM-based HPC cluster.
- `slurm_utils.py` (included in `scripts/`) handles job submission and
  monitoring. The generated SLURM scripts contain cluster-specific
  settings (partition names, `wckey`, module load commands, MPI launcher
  flags) that were used on Teton — **you will need to adapt these for
  your own cluster** before running.

### Run the optimization

The script is called from any directory; outputs are written to
`{material}/sigma5_310_STGB/` relative to the `density_optimization/`
root:

```bash
python scripts/optimize.py --material Fe
python scripts/optimize.py --material Ni
```

Outputs written on completion:

| File | Description |
|------|-------------|
| `{material}/sigma5_310_STGB/density_gbe_log.json` | Log of (normalized density, GBE) for every successful evaluation |
| `{material}/sigma5_310_STGB/best_density_structure.txt` | LAMMPS dump of the highest-density relaxed structure |
| `{material}/sigma5_310_STGB/initial.dat` | Initial LAMMPS structure used as the starting point |

Intermediate LAMMPS jobs run in `{material}/sigma5_310_STGB/workdir.1/`
and are cleaned up automatically after each generation.

### Optimization parameters

The following parameters were used in the manuscript and are hardcoded
in `optimize.py`:

| Parameter | Value |
|-----------|-------|
| Population size | 50 |
| Generations | 200 |
| Random seed | 0 |
| Repeat factor | (5, 9) |
| Moves | insert_atoms, remove_atoms, translate_right_grain |

| Material | Lattice parameter | Structure | Interaction distance |
|----------|-------------------|-----------|----------------------|
| Fe | 2.855 Å | BCC | 5.3 Å |
| Ni | 3.52 Å | FCC | 4.85 Å |
