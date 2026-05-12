from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from GBOpt import GBMaker, GBManipulator, GBMinimizer

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
LAMMPS_DIR = Path(os.environ.get("LAMMPS_DIR", PROJECT_ROOT / "lammps"))


def _is_teton() -> bool:
    cluster = os.environ.get("CLUSTER", "")
    if cluster:
        return cluster.lower() == "teton"
    return "teton" in socket.gethostname().lower()


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_boundaries():
    spec = importlib.util.spec_from_file_location(
        "boundaries", PROJECT_ROOT / "config" / "boundaries.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BOUNDARIES


def get_gb_energy(
    GB: GBMaker,
    manipulator: GBManipulator,
    atom_positions: np.ndarray,
    unique_id: str = "",
    *,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    n_threads: int = 1,
    srun_mpi: str = "none",
    wckey: str = "",
    time: str = "",
    ntasks: int = 1,
    cpus: int = 1,
    nodes: int = 1,
    partition: str = "",
    module: str = "",
    material: str = "",
    **kwargs,
) -> Tuple[float, str]:
    box_dims = manipulator.parents[0].box_dims
    out_structure = f"output_{unique_id}.dat"
    output_txt = f"output_{unique_id}.txt"
    results_out = f"results_{unique_id}.txt"
    logfile = f"log_{unique_id}.lammps"

    GB.write_lammps(out_structure, atom_positions, box_dims, type_as_int=True)

    input_template = Path(input_script)
    min_strat = input_template.parent / "min_strat.commands"
    if min_strat.is_file():
        shutil.copy(min_strat, Path.cwd() / "min_strat.commands")

    potential_setup = input_template.parent / material / "potential.setup"
    if potential_setup.is_file():
        shutil.copy(potential_setup, Path.cwd() / "potential.setup")

    addtl_args = " ".join(f"-var {k} {v}" for k, v in kwargs.items())

    if _is_teton():
        kokkos = f"-k on t {n_threads} -sf kk "
    else:
        kokkos = ""

    srun_flags = f"--mpi={srun_mpi} --hint=nomultithread"
    srun_flags += f" --ntasks-per-node={ntasks} --cpus-per-task={cpus} --nodes={nodes}"
    if time:
        srun_flags += f" --time={time}"
    if wckey:
        srun_flags += f" --wckey {wckey}"
    if partition:
        srun_flags += f" -p {partition}"

    lammps_cmd = (
        f"srun {srun_flags} {lmp_binary} {kokkos}"
        f"-in {input_script} {addtl_args} "
        f"-var datafile {out_structure} -var unique_id {unique_id} "
        f"-log {logfile} -sc {output_txt}"
    )
    cmd = f"ml {module} && {lammps_cmd}" if module else lammps_cmd
    print(f"Running:\n  {cmd}")

    env = os.environ.copy()
    if _is_teton():
        env["OMP_NUM_THREADS"] = str(n_threads)

    with open(results_out, "w") as f:
        P = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           env=env, check=False, shell=True)

    if not P.returncode:
        txt = Path(results_out).read_text()
        gbe_val = float(txt.split()[0])
        for p in (out_structure, output_txt, results_out):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        return gbe_val, f"final_{unique_id}.dump"
    else:
        return 1.0e30, ""


def run_mc(
    *,
    lattice_parameter: float,
    structure: str,
    atom_types: str,
    interaction_distance: float,
    x_dim_min: int = 60,
    repeat_factor: Tuple[int, int] = (2, 3),
    misorientation: np.ndarray,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    max_steps: int = 500,
    seed: int = 123,
    cooldown_rate: float = 0.99,
    e_tol: float = 1e-8,
    e_accept: float = 0.1,
    choices: List[str] | None = None,
    slurm_cfg: dict | None = None,
    material: str = "",
    initial_structure: Path | None = None,
) -> None:
    if choices is None:
        choices = ["translate_right_grain", "remove_atoms", "insert_atoms"]

    slurm = slurm_cfg or {}
    n_threads = slurm.get("omp_num_threads") or slurm.get("cpus_per_task", 1)
    srun_mpi = slurm.get("srun_mpi", "none")
    wckey = slurm.get("wckey", "")
    time = slurm.get("time", "")
    ntasks = slurm.get("ntasks_per_node", 1)
    cpus = slurm.get("cpus_per_task", 1)
    nodes = slurm.get("nodes", 1)
    partition = slurm.get("partition", "")
    module = slurm.get("module", "")

    energy_fn = partial(
        get_gb_energy,
        lmp_binary=lmp_binary,
        input_script=input_script,
        n_threads=n_threads,
        srun_mpi=srun_mpi,
        wckey=wckey,
        time=time,
        ntasks=ntasks,
        cpus=cpus,
        nodes=nodes,
        partition=partition,
        module=module,
        material=material
    )

    GB0 = GBMaker(
        lattice_parameter, structure, lattice_parameter, misorientation,
        atom_types=atom_types,
        interaction_distance=interaction_distance,
        x_dim_min=x_dim_min,
        repeat_factor=repeat_factor,
        vacuum=0,
    )
    gb_thickness = 2 * max(GB0.spacing["x"]["left"], GB0.spacing["x"]["right"])
    del GB0

    GB = GBMaker(
        lattice_parameter, structure, gb_thickness, misorientation,
        atom_types=atom_types,
        interaction_distance=interaction_distance,
        x_dim_min=x_dim_min,
        repeat_factor=repeat_factor,
        vacuum=0,
    )

    extra = {"initial_structure": str(
        initial_structure)} if initial_structure is not None else {}
    MC = GBMinimizer.MonteCarloMinimizer(
        copy.deepcopy(GB),
        energy_fn,
        choices=choices,
        seed=seed,
        **extra,
    )

    MC.run_MC(
        max_steps=max_steps,
        unique_id=1,
        cooldown_rate=cooldown_rate,
        E_tol=e_tol,
        E_accept=e_accept,
        # checkpoint_file="checkpoint.json",
    )

    op_list = MC.operation_list
    with open("gbe_vals.txt", "w") as f:
        for i, val in enumerate(MC.GBE_vals):
            f.write(f"{op_list[i][0]} {val} {op_list[i][1]}\n")

    print(f"Final minimum GBE = {min(MC.GBE_vals)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MC grain boundary optimization")
    parser.add_argument("--material", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--run", type=int, default=1, metavar="N",
                        help="Run index; controls output directory and seed offset (seed = base_seed + N - 1)")
    parser.add_argument("--high-energy", action="store_true",
                        help="Use high-energy restart structure from materials/")
    parser.add_argument("--e-accept", type=float, default=None, metavar="E",
                        help="Acceptance energy threshold in eV (overrides mc.toml)")
    parser.add_argument("--initial-structure", default=None, metavar="PATH",
                        help="Path to initial LAMMPS data file; overrides GBMaker-generated structure")
    args = parser.parse_args()

    init_type = "high_energy" if args.high_energy else "standard"
    run_dir = PROJECT_ROOT / args.material / args.boundary / "MC" / init_type / f"run{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    mat_path = PROJECT_ROOT / "materials" / f"{args.material}.toml"
    with open(mat_path, "rb") as fh:
        mat = tomllib.load(fh)

    mc_path = PROJECT_ROOT / "config" / "mc.toml"
    with open(mc_path, "rb") as fh:
        mc_cfg = tomllib.load(fh)

    overrides_path = Path("overrides.toml")
    if overrides_path.is_file():
        with open(overrides_path, "rb") as fh:
            mc_cfg = _deep_merge(mc_cfg, tomllib.load(fh))

    if args.e_accept is not None:
        mc_cfg["e_accept"] = args.e_accept

    boundaries = _load_boundaries()
    if args.boundary not in boundaries:
        print(
            f"Unknown boundary '{args.boundary}'. Available: {list(boundaries)}", file=sys.stderr)
        sys.exit(1)

    bnd = boundaries[args.boundary]
    repeat_factor = tuple(mat.get("repeat_factor", bnd["repeat_factor"]))

    if args.initial_structure is not None:
        initial_structure = Path(args.initial_structure).resolve()
    elif args.high_energy:
        he_path = PROJECT_ROOT / args.material / args.boundary / "high_e_initial.txt"
        initial_structure = he_path if he_path.is_file() else None
    elif mc_cfg.get("initial_structure"):
        initial_structure = PROJECT_ROOT / mc_cfg["initial_structure"]
    else:
        initial_structure = None

    seed = mc_cfg.get("seed", 123) + (args.run - 1)

    run_mc(
        lattice_parameter=mat["lattice_parameter"],
        structure=mat["structure"],
        atom_types=mat["atom_types"],
        interaction_distance=mat["interaction_distance"],
        x_dim_min=mat.get("x_dim_min", 60),
        repeat_factor=repeat_factor,
        misorientation=bnd["misorientation"],
        lmp_binary=mc_cfg.get("lmp_binary", "lmp"),
        input_script=mc_cfg.get("input_script", str(LAMMPS_DIR / "lmp.in")),
        max_steps=mc_cfg.get("max_steps", 500),
        seed=seed,
        cooldown_rate=mc_cfg.get("cooldown_rate", 0.99),
        e_tol=mc_cfg.get("e_tol", 1e-8),
        e_accept=mc_cfg.get("e_accept", 0.1),
        choices=mc_cfg.get("choices"),
        slurm_cfg=mc_cfg.get("slurm"),
        material=args.material,
        initial_structure=initial_structure,
    )


if __name__ == "__main__":
    main()
