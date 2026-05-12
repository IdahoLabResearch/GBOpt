from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import shutil
import socket
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from slurm_utils import SlurmJob, submit_job, wait_for_jobs

from GBOpt import GBMaker, GBManipulator, GBMinimizer

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SLURM_WORK_ROOT = Path.cwd()
LAMMPS_DIR = Path(os.environ.get("LAMMPS_DIR", PROJECT_ROOT / "lammps"))

ERROR_SIGNATURES = {
    "lost_atoms": "ERROR: Lost atoms",
    "non_numeric_atom_coords": "ERROR: Non-numeric atom coords",
    "non_numeric_pressure": "ERROR: Non-numeric pressure",
    "non_numeric_box_dimensions": "ERROR: Non-numeric box dimensions",
    "non_numeric_unstable": "ERROR: Non-numeric",
}
PENALTY = 1.0e30


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


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(errors="ignore")
    except Exception:
        return False


def _detect_failure_reason(paths: Sequence[Path]) -> str | None:
    ordered = [
        ("lost_atoms", ERROR_SIGNATURES["lost_atoms"]),
        ("non_numeric_atom_coords", ERROR_SIGNATURES["non_numeric_atom_coords"]),
        ("non_numeric_pressure", ERROR_SIGNATURES["non_numeric_pressure"]),
        ("non_numeric_box_dimensions", ERROR_SIGNATURES["non_numeric_box_dimensions"]),
    ]
    for reason, sig in ordered:
        if any(_file_contains(p, sig) for p in paths):
            return reason
    if any(_file_contains(p, ERROR_SIGNATURES["non_numeric_unstable"]) for p in paths):
        return "non_numeric_unstable"
    return None


def _wait_for_fresh_file(
    path: Path, min_mtime: float, timeout_s: float = 60.0, poll_s: float = 0.25
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            st = path.stat()
            if st.st_mtime >= (min_mtime - 1.0):
                return True
        except FileNotFoundError:
            pass
        try:
            list(path.parent.iterdir())
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def _parse_unique_id(unique_id: str) -> tuple[str, str, str]:
    if unique_id.startswith("GA_initial"):
        return unique_id[len("GA_initial"):], "initial", "0"
    if unique_id.startswith("GA_") and "_g" in unique_id and "_c" in unique_id:
        prefix, rest = unique_id.split("_g", 1)
        run_uuid = prefix[len("GA_"):]
        gen_str, cand_str = rest.split("_c", 1)
        return run_uuid, gen_str, cand_str
    return unique_id, "initial", "0"


def get_gb_energy(
    gb: GBMaker,
    manipulator: GBManipulator,
    atom_positions: np.ndarray,
    unique_id: str,
    *,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    slurm_cfg: dict | None = None,
    material: str = "",
    **kwargs,
) -> Tuple[float, str]:
    batch_results = evaluate_batch(
        gb,
        manipulators=[manipulator],
        candidates=[atom_positions],
        lineages=[[unique_id]],
        unique_ids=[unique_id],
        lmp_binary=lmp_binary,
        input_script=input_script,
        slurm_cfg=slurm_cfg,
        material=material,
        **kwargs,
    )
    result = batch_results[0]
    return result["energy"], result["final_dump"]


def evaluate_batch(
    gb: GBMaker,
    manipulators: Sequence[GBManipulator],
    candidates: Sequence[np.ndarray],
    lineages: Sequence[List[str]],
    unique_ids: Sequence[str | int],
    *,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    slurm_cfg: dict | None = None,
    material: str = "",
    **kwargs,
) -> List[Dict[str, Any]]:
    if not (len(candidates) == len(unique_ids) == len(manipulators) == len(lineages)):
        raise ValueError(
            "candidates, unique_ids, manipulators, and lineages must have equal length."
        )

    input_template = Path(input_script)
    if not input_template.is_file():
        raise FileNotFoundError(f"LAMMPS input template not found: {input_template}")

    slurm = slurm_cfg or {}
    s_time = slurm.get("time", "1:00:00")
    s_ntasks = slurm.get("ntasks_per_node", 1)
    s_cpus = slurm.get("cpus_per_task", 1)
    s_nodes = slurm.get("nodes", 1)
    s_wckey = slurm.get("wckey", "")
    s_partition = slurm.get("partition", "short")
    s_module = slurm.get("module")
    s_omp = slurm.get("omp_num_threads")
    s_srun_mpi = slurm.get("srun_mpi", "none")
    use_kokkos = _is_teton()
    n_kokkos_threads = s_omp if s_omp is not None else s_cpus

    jobs: List[SlurmJob] = []
    submit_ts: Dict[str, float] = {}
    uid_to_gendir: Dict[str, Path] = {}

    for manip, cand, lineage, uid in zip(manipulators, candidates, lineages, unique_ids):
        uid_str = str(uid)
        run_uuid, gen_str, cand_str = _parse_unique_id(uid_str)

        run_dir = SLURM_WORK_ROOT / f"workdir.{run_uuid}"
        gen_dir = run_dir / ("initial" if gen_str == "initial" else f"gen_{gen_str}")
        gen_dir.mkdir(parents=True, exist_ok=True)

        box_dims = manip.parents[0].box_dims
        temp_dump = gen_dir / f"temp_{uid_str}.dump"
        gb.write_lammps(str(temp_dump), cand, box_dims, type_as_int=True)

        cand_input = gen_dir / f"lmp_c{cand_str}.in"
        shutil.copy(input_template, cand_input)
        min_strat = input_template.parent / "min_strat.commands"
        if min_strat.is_file():
            shutil.copy(min_strat, gen_dir / "min_strat.commands")

        potential = input_template.parent / material / "potential.setup"
        if potential.is_file():
            shutil.copy(potential, gen_dir / "potential.setup")

        sbatch_lines = [
            f"#SBATCH -D {gen_dir}",
            f"#SBATCH --time={s_time}",
            f"#SBATCH --ntasks-per-node={s_ntasks}",
            f"#SBATCH --cpus-per-task={s_cpus}",
            f"#SBATCH --nodes={s_nodes}",
        ]
        if s_wckey:
            sbatch_lines.append(f"#SBATCH --wckey {s_wckey}")
        if s_partition:
            sbatch_lines.append(f"#SBATCH -p {s_partition}")
        sbatch_lines += [
            f'#SBATCH -J "lammps_job_{uid_str}"',
            f"#SBATCH -o slurm_{uid_str}.out",
            f"#SBATCH -e slurm_{uid_str}.err",
        ]

        kokkos_flags = f"-k on t {n_kokkos_threads} -sf kk \\\n    " if use_kokkos else ""
        lammps_cmd = (
            f"srun --mpi={s_srun_mpi} --hint=nomultithread {lmp_binary} {kokkos_flags}"
            f"-in {cand_input.name} \\\n"
            f"    -var datafile {temp_dump} \\\n"
            f"    -var unique_id {uid_str} \\\n"
            f"    -log log_{uid_str}.lammps \\\n"
            f"    > output_{uid_str}.txt 2>&1"
        )

        script_lines = ["#!/bin/bash"] + sbatch_lines + [""]
        if s_module:
            script_lines.append(f"ml {s_module}")
        if use_kokkos:
            script_lines.append(f"export OMP_NUM_THREADS={n_kokkos_threads}")
        script_lines += [
            "",
            f"cd {gen_dir}",
            "",
            f'echo "Running LAMMPS for unique_id={uid_str} in $(pwd)"',
            "",
            lammps_cmd,
            "",
            "rc=$?",
            "",
            f'if [ ! -f "results_{uid_str}.txt" ]; then',
            f'  echo "1.0e30" > "results_{uid_str}.txt"',
            "fi",
            "",
            "sync",
            "",
        ]

        slurm_script = gen_dir / f"slurm_{uid_str}.sh"
        slurm_script.write_text("\n".join(script_lines))
        slurm_script.chmod(0o750)

        for p in (
            gen_dir / f"results_{uid_str}.txt",
            gen_dir / f"final_{uid_str}.dump",
            gen_dir / f"output_{uid_str}.txt",
            gen_dir / f"log_{uid_str}.lammps",
            gen_dir / f"slurm_{uid_str}.out",
            gen_dir / f"slurm_{uid_str}.err",
        ):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        submit_ts[uid_str] = time.time()
        uid_to_gendir[uid_str] = gen_dir

        job = submit_job(slurm_script)
        print(f"Submitted SLURM job {job.job_id} for unique_id={uid_str}")
        jobs.append(job)

    print(f"Waiting for {len(jobs)} SLURM jobs to finish...")
    wait_for_jobs(jobs, poll_interval=1.0, fail_on=())
    print("All SLURM jobs finished; collecting results.")

    results: List[Dict[str, Any]] = []

    for i, uid in enumerate(unique_ids):
        uid_str = str(uid)
        run_uuid, gen_str, _ = _parse_unique_id(uid_str)

        run_dir = SLURM_WORK_ROOT / f"workdir.{run_uuid}"
        gen_dir = run_dir / ("initial" if gen_str == "initial" else f"gen_{gen_str}")

        results_out = gen_dir / f"results_{uid_str}.txt"
        final_dump = gen_dir / f"final_{uid_str}.dump"
        output_txt = gen_dir / f"output_{uid_str}.txt"
        logfile = gen_dir / f"log_{uid_str}.lammps"
        temp_dump = gen_dir / f"temp_{uid_str}.dump"

        min_mtime = submit_ts.get(uid_str, time.time())

        if not _wait_for_fresh_file(results_out, min_mtime=min_mtime, timeout_s=60.0, poll_s=0.25):
            reason = _detect_failure_reason([logfile, output_txt])
            if reason is not None:
                results.append({
                    "energy": PENALTY,
                    "final_dump": None,
                    "num_atoms": int(candidates[i].shape[0]),
                    "parents": list(lineages[i]),
                    "status": "failed",
                    "fail_reason": reason,
                })
                continue

            try:
                listing = "\n".join(sorted(p.name for p in gen_dir.iterdir()))
            except Exception as e:
                listing = f"<failed to list {gen_dir}: {e}>"

            raise RuntimeError(
                f"Expected fresh results file not found for unique_id={uid_str}: {results_out}\n"
                f"min_mtime={min_mtime} (epoch seconds)\n"
                f"Directory listing of {gen_dir}:\n{listing}\n"
                f"slurm out: {gen_dir / f'slurm_{uid_str}.out'}\n"
                f"slurm err: {gen_dir / f'slurm_{uid_str}.err'}\n"
                f"output txt: {gen_dir / f'output_{uid_str}.txt'}\n"
            )

        txt = results_out.read_text().strip()
        parts = txt.split()
        gbe_val = float(parts[0])
        reason = " ".join(parts[1:]) if len(parts) > 1 else None

        status = "ok" if gbe_val < PENALTY else "failed"

        record = {
            "energy": float(gbe_val),
            "final_dump": str(final_dump),
            "num_atoms": int(candidates[i].shape[0]),
            "parents": list(lineages[i]),
            "status": status,
        }
        if status != "ok":
            record["fail_reason"] = reason or "penalty"
        results.append(record)

        if status == "ok":
            for p in (results_out, output_txt, logfile, temp_dump):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    return results


def run_evolution(
    *,
    lattice_parameter: float,
    structure: str,
    atom_types: str | List[str],
    interaction_distance: float,
    x_dim_min: int = 60,
    repeat_factor: Tuple[int, int] = (2, 3),
    misorientation: np.ndarray,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    population_size: int = 50,
    generations: int = 200,
    seed: int = 0,
    choices: List[str] | None = None,
    slurm_cfg: dict | None = None,
    material: str = "",
    initial_structure: Path | None = None,
) -> None:
    if choices is None:
        choices = ["insert_atoms", "remove_atoms", "translate_right_grain"]

    gb_energy = partial(
        get_gb_energy,
        lmp_binary=lmp_binary,
        input_script=input_script,
        slurm_cfg=slurm_cfg,
        material=material,
    )
    gb_batch_energy = partial(
        evaluate_batch,
        lmp_binary=lmp_binary,
        input_script=input_script,
        slurm_cfg=slurm_cfg,
        material=material,
    )

    GB0 = GBMaker(
        lattice_parameter, structure, lattice_parameter, misorientation,
        atom_types=atom_types,
        x_dim_min=x_dim_min,
        repeat_factor=repeat_factor,
        interaction_distance=interaction_distance,
        vacuum=0,
    )
    gb_thickness = 2 * max(GB0.spacing["x"]["left"], GB0.spacing["x"]["right"])
    del GB0

    GB = GBMaker(
        lattice_parameter, structure, gb_thickness, misorientation,
        atom_types=atom_types,
        x_dim_min=x_dim_min,
        repeat_factor=repeat_factor,
        interaction_distance=interaction_distance,
        vacuum=0,
    )

    if initial_structure is not None:
        shutil.copy(initial_structure, "initial.dat")
    else:
        GB.write_lammps("initial.dat", type_as_int=True)

    extra = {"initial_structure": str(
        initial_structure)} if initial_structure is not None else {}
    ga_minimizer = GBMinimizer.GeneticAlgorithmMinimizer(
        GB,
        gb_energy_func=gb_energy,
        choices=choices,
        seed=seed,
        population_size=population_size,
        generations=generations,
        gb_batch_energy_func=gb_batch_energy,
        **extra,
    )

    min_gbe, _ = ga_minimizer.run_GA(
        unique_id=1,
        # checkpoint_file="checkpoint.json",
    )
    print(f"Final minimum GBE = {min_gbe}")

    with open("pickled_ga_object.pkl", "wb") as f:
        pickle.dump(ga_minimizer, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA grain boundary optimization")
    parser.add_argument("--material", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--run", type=int, default=1, metavar="N",
                        help="Run index; controls output directory and seed offset (seed = base_seed + N - 1)")
    parser.add_argument("--high-energy", action="store_true",
                        help="Use high-energy restart structure from materials/")
    parser.add_argument("--initial-structure", default=None, metavar="PATH",
                        help="Path to initial LAMMPS data file; overrides GBMaker-generated structure")
    args = parser.parse_args()

    init_type = "high_energy" if args.high_energy else "standard"
    run_dir = PROJECT_ROOT / args.material / args.boundary / "GA" / init_type / f"run{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    global SLURM_WORK_ROOT
    SLURM_WORK_ROOT = run_dir

    mat_path = PROJECT_ROOT / "materials" / f"{args.material}.toml"
    with open(mat_path, "rb") as fh:
        mat = tomllib.load(fh)

    ga_path = PROJECT_ROOT / "config" / "ga.toml"
    with open(ga_path, "rb") as fh:
        ga_cfg = tomllib.load(fh)

    overrides_path = Path("overrides.toml")
    if overrides_path.is_file():
        with open(overrides_path, "rb") as fh:
            ga_cfg = _deep_merge(ga_cfg, tomllib.load(fh))

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
    elif ga_cfg.get("initial_structure"):
        initial_structure = PROJECT_ROOT / ga_cfg["initial_structure"]
    else:
        initial_structure = None

    seed = ga_cfg.get("seed", 0) + (args.run - 1)

    run_evolution(
        lattice_parameter=mat["lattice_parameter"],
        structure=mat["structure"],
        atom_types=mat["atom_types"],
        interaction_distance=mat["interaction_distance"],
        x_dim_min=mat.get("x_dim_min", 60),
        repeat_factor=repeat_factor,
        misorientation=bnd["misorientation"],
        lmp_binary=ga_cfg.get("lmp_binary", "lmp"),
        input_script=ga_cfg.get("input_script", str(LAMMPS_DIR / "lmp.in")),
        population_size=ga_cfg.get("population_size", 50),
        generations=ga_cfg.get("generations", 200),
        seed=seed,
        choices=ga_cfg.get("choices"),
        slurm_cfg=ga_cfg.get("slurm"),
        material=args.material,
        initial_structure=initial_structure,
    )


if __name__ == "__main__":
    main()
