"""
optimize.py
-----------
Maximize the density of GB core atoms for a Σ5[001]{310} GB using
GeneticAlgorithmMinimizer.

Optimization metric  : raw GB core atom count (negated, so the GA minimizes)
Reporting metric     : n_gb_core / n_perfect  (dimensionless, 2D planar reference)

Outputs (written to {density_optimization}/{material}/sigma5_310_STGB/)
-------
density_gbe_log.json         list of [normalized_density, gbe_J_m2] per evaluation
best_density_structure.dump  LAMMPS dump of the highest-density relaxed structure
initial.dat                  initial LAMMPS structure

Usage (from any directory):
    python scripts/optimize.py --material Fe
    python scripts/optimize.py --material Ni
"""

import argparse
import json
import math
import shutil
import time
from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from slurm_utils import SlurmJob, submit_job, wait_for_jobs

from GBOpt import GBMaker, GBManipulator, GBMinimizer

PENALTY = 1.0e30
BOUNDARY = "sigma5_310_STGB"

MATERIAL_PARAMS = {
    "Fe": dict(lattice_parameter=2.855, structure="bcc",
               atom_types="Fe", interaction_distance=5.3),
    "Ni": dict(lattice_parameter=3.52,  structure="fcc",
               atom_types="Ni", interaction_distance=4.85),
}

ERROR_SIGNATURES = {
    "lost_atoms":                "ERROR: Lost atoms",
    "non_numeric_atom_coords":   "ERROR: Non-numeric atom coords",
    "non_numeric_pressure":      "ERROR: Non-numeric pressure",
    "non_numeric_box_dimensions": "ERROR: Non-numeric box dimensions",
    "non_numeric_unstable":      "ERROR: Non-numeric",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(errors="ignore")
    except Exception:
        return False


def _detect_failure_reason(paths: Sequence[Path]) -> str | None:
    ordered = [
        ("lost_atoms",                 ERROR_SIGNATURES["lost_atoms"]),
        ("non_numeric_atom_coords",    ERROR_SIGNATURES["non_numeric_atom_coords"]),
        ("non_numeric_pressure",       ERROR_SIGNATURES["non_numeric_pressure"]),
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
            if path.stat().st_mtime >= (min_mtime - 1.0):
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
        gen_str, cand_str = rest.split("_c", 1)
        return prefix[len("GA_"):], gen_str, cand_str
    return unique_id, "initial", "0"


# ---------------------------------------------------------------------------
# Density helper
# ---------------------------------------------------------------------------

def _unwrap_candidate(cand) -> np.ndarray:
    """Mutator.mutate() returns (label, array); carried-over structures are plain arrays."""
    if isinstance(cand, tuple):
        return cand[1]
    return cand


def _count_gb_core_atoms(
    atom_positions: np.ndarray, box_dims: np.ndarray, gb_thickness: float
) -> int:
    """Count atoms within the GB core slab (±gb_thickness/2 of the box x-centre)."""
    x_center = box_dims[0, 1] / 2
    half_t = gb_thickness / 2
    return int(np.sum(
        (atom_positions["x"] >= x_center - half_t) &
        (atom_positions["x"] <= x_center + half_t)
    ))


# ---------------------------------------------------------------------------
# LAMMPS evaluation
# ---------------------------------------------------------------------------

def get_gb_energy(
    gb: GBMaker,
    manipulator: GBManipulator,
    atom_positions: np.ndarray,
    unique_id: str,
    *,
    work_dir: Path,
    omp_threads: int = 28,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    job_walltime: str = "06:00:00",
    **kwargs,
) -> Tuple[float, str]:
    batch_results = evaluate_batch(
        gb,
        manipulators=[manipulator],
        candidates=[atom_positions],
        lineages=[[unique_id]],
        unique_ids=[unique_id],
        work_dir=work_dir,
        omp_threads=omp_threads,
        lmp_binary=lmp_binary,
        input_script=input_script,
        job_walltime=job_walltime,
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
    work_dir: Path,
    omp_threads: int = 28,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    job_walltime: str = "06:00:00",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Submit one SLURM/KOKKOS LAMMPS job per candidate and collect GBE results."""
    if not (len(candidates) == len(unique_ids) == len(manipulators) == len(lineages)):
        raise ValueError(
            "candidates, unique_ids, manipulators, and lineages must have equal length."
        )

    input_template = Path(input_script)
    if not input_template.is_file():
        raise FileNotFoundError(f"LAMMPS input template not found: {input_template}")

    jobs: List[SlurmJob] = []
    submit_ts: Dict[str, float] = {}

    for manip, cand, lineage, uid in zip(manipulators, candidates, lineages, unique_ids):
        uid_str = str(uid)
        run_uuid, gen_str, cand_str = _parse_unique_id(uid_str)

        run_dir = work_dir / f"workdir.{run_uuid}"
        gen_dir = run_dir / ("initial" if gen_str == "initial" else f"gen_{gen_str}")
        gen_dir.mkdir(parents=True, exist_ok=True)

        box_dims = manip.parents[0].box_dims
        temp_dump = gen_dir / f"temp_{uid_str}.dump"
        gb.write_lammps(str(temp_dump), _unwrap_candidate(
            cand), box_dims, type_as_int=True)

        cand_input = gen_dir / f"lmp_c{cand_str}.in"
        shutil.copy(input_template, cand_input)

        slurm_script = gen_dir / f"slurm_{uid_str}.sh"
        slurm_script.write_text(dedent(f"""\
            #!/bin/bash
            #SBATCH -D {gen_dir}
            #SBATCH --time={job_walltime}
            #SBATCH --partition=short
            #SBATCH --ntasks-per-node=1
            #SBATCH --cpus-per-task={omp_threads}
            #SBATCH --nodes=1
            #SBATCH --wckey ne_ldrd
            #SBATCH -J "density_lmp_{uid_str}"
            #SBATCH -o slurm_{uid_str}.out
            #SBATCH -e slurm_{uid_str}.err

            ml lammps-user

            cd {gen_dir}

            echo "Running LAMMPS for unique_id={uid_str} in $(pwd)"

            export OMP_NUM_THREADS={omp_threads}
            srun --mpi=cray_shasta --hint=nomultithread \\
                {lmp_binary} -k on t {omp_threads} -sf kk \\
                -in {cand_input.name} \\
                -var datafile {temp_dump} \\
                -var unique_id {uid_str} \\
                -log log_{uid_str}.lammps \\
                > output_{uid_str}.txt 2>&1

            if [ ! -f "results_{uid_str}.txt" ]; then
              echo "1.0e30" > "results_{uid_str}.txt"
            fi

            sync
            """))
        slurm_script.chmod(0o750)

        for p in (
            gen_dir / f"results_{uid_str}.txt",
            gen_dir / f"final_{uid_str}.dump",
            gen_dir / f"output_{uid_str}.txt",
            gen_dir / f"log_{uid_str}.lammps",
            gen_dir / f"slurm_{uid_str}.out",
            gen_dir / f"slurm_{uid_str}.err",
        ):
            p.unlink(missing_ok=True)

        submit_ts[uid_str] = time.time()
        job = submit_job(slurm_script)
        print(f"Submitted SLURM job {job.job_id} for unique_id={uid_str}")
        jobs.append(job)

    print(f"Waiting for {len(jobs)} SLURM jobs...")
    wait_for_jobs(jobs, poll_interval=1.0)
    print("All jobs finished; collecting results.")

    results: List[Dict[str, Any]] = []

    for i, uid in enumerate(unique_ids):
        uid_str = str(uid)
        run_uuid, gen_str, _ = _parse_unique_id(uid_str)
        run_dir = work_dir / f"workdir.{run_uuid}"
        gen_dir = run_dir / ("initial" if gen_str == "initial" else f"gen_{gen_str}")

        results_out = gen_dir / f"results_{uid_str}.txt"
        final_dump = gen_dir / f"final_{uid_str}.dump"
        output_txt = gen_dir / f"output_{uid_str}.txt"
        logfile = gen_dir / f"log_{uid_str}.lammps"
        temp_dump = gen_dir / f"temp_{uid_str}.dump"

        min_mtime = submit_ts.get(uid_str, time.time())

        if not _wait_for_fresh_file(results_out, min_mtime=min_mtime,
                                    timeout_s=60.0, poll_s=0.25):
            reason = _detect_failure_reason([logfile, output_txt])
            if reason is not None:
                results.append({
                    "energy":    PENALTY, "final_dump": None,
                    "num_atoms": int(len(_unwrap_candidate(candidates[i]))),
                    "parents":   list(lineages[i]),
                    "status":    "failed", "fail_reason": reason,
                })
                continue
            try:
                listing = "\n".join(sorted(p.name for p in gen_dir.iterdir()))
            except Exception as e:
                listing = f"<failed to list {gen_dir}: {e}>"
            raise RuntimeError(
                f"Results file not found for unique_id={uid_str}: {results_out}\n"
                f"Directory listing:\n{listing}\n"
            )

        txt = results_out.read_text().strip().split()
        gbe_val = float(txt[0])
        status = "ok" if gbe_val < PENALTY else "failed"
        record = {
            "energy":    gbe_val,
            "final_dump": str(final_dump),
            "num_atoms": int(len(candidates[i])),
            "parents":   list(lineages[i]),
            "status":    status,
        }
        if status != "ok":
            record["fail_reason"] = " ".join(txt[1:]) or "penalty"
        results.append(record)

        for p in (results_out, output_txt, logfile, temp_dump):
            p.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------

def run_density_optimization(
    *,
    lattice_parameter: float,
    structure: str,
    atom_types: str | List[str],
    interaction_distance: float,
    output_dir: Path,
    work_dir: Path,
    x_dim_min: int = 60,
    repeat_factor: Tuple[int, int] = (5, 9),
    omp_threads: int = 28,
    lmp_binary: str = "lmp",
    input_script: str = "lmp.in",
    job_walltime: str = "06:00:00",
    population_size: int = 50,
    generations: int = 200,
    seed: int = 0,
    initial_structure: str | None = None,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- GB geometry ---
    theta = math.radians(36.869898)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])

    GB0 = GBMaker(
        lattice_parameter, structure, lattice_parameter, misorientation, atom_types,
        x_dim_min=x_dim_min, repeat_factor=repeat_factor,
        interaction_distance=interaction_distance, vacuum=0,
    )
    gb_thickness = 2 * max(GB0.spacing["x"]["left"], GB0.spacing["x"]["right"])
    del GB0

    GB = GBMaker(
        lattice_parameter, structure, gb_thickness, misorientation, atom_types,
        x_dim_min=x_dim_min, repeat_factor=repeat_factor,
        interaction_distance=interaction_distance, vacuum=0,
    )
    GB.write_lammps(str(output_dir / "initial.dat"), type_as_int=True)

    # --- Perfect 2D planar reference ---
    n_perfect = (GB.y_dim / GB.spacing["y"]) * (GB.z_dim / GB.spacing["z"])
    print(f"GB plane: y={GB.y_dim:.3f} Å  z={GB.z_dim:.3f} Å  "
          f"spacing_y={GB.spacing['y']:.3f} Å  spacing_z={GB.spacing['z']:.3f} Å")
    print(f"Perfect plane reference count: {n_perfect:.2f}")

    # --- Logging ---
    density_gbe_log: List[Tuple[float, float]] = []

    _lmp_kwargs = dict(
        work_dir=work_dir,
        omp_threads=omp_threads,
        lmp_binary=lmp_binary,
        input_script=input_script,
        job_walltime=job_walltime,
    )
    _single = partial(get_gb_energy,  **_lmp_kwargs)
    _batch = partial(evaluate_batch, **_lmp_kwargs)

    def get_density_single(gb, manipulator, atom_positions, unique_id, **kw):
        """Single-candidate wrapper: returns (-n_gb_core, dump_file)."""
        box_dims = manipulator.parents[0].box_dims
        n_gb_core = _count_gb_core_atoms(atom_positions, box_dims, gb.gb_thickness)
        gbe, dump_file = _single(gb, manipulator, atom_positions, unique_id, **kw)
        if gbe < PENALTY:
            density_gbe_log.append((n_gb_core / n_perfect, float(gbe)))
        return -n_gb_core, dump_file

    def get_density_batch(gb, manipulators, candidates, lineages, unique_ids, **kw):
        """Batch wrapper: computes core counts, runs LAMMPS, patches energies."""
        core_counts = [
            _count_gb_core_atoms(
                _unwrap_candidate(cand), manip.parents[0].box_dims, gb.gb_thickness
            )
            for manip, cand in zip(manipulators, candidates)
        ]
        results = _batch(gb, manipulators, candidates, lineages, unique_ids, **kw)
        for count, result in zip(core_counts, results):
            if result["energy"] < PENALTY:
                density_gbe_log.append((count / n_perfect, float(result["energy"])))
            result["energy"] = -count
        return results

    # --- GA ---
    ga = GBMinimizer.GeneticAlgorithmMinimizer(
        GB,
        gb_energy_func=get_density_single,
        choices=["insert_atoms", "remove_atoms", "translate_right_grain"],
        seed=seed,
        population_size=population_size,
        generations=generations,
        gb_batch_energy_func=get_density_batch,
        initial_structure=initial_structure,
    )

    best_neg_count, best_dump = ga.run_GA(
        unique_id=1,
        # checkpoint_file="checkpoint.json",
    )
    best_n_gb_core = -best_neg_count
    best_normalized = best_n_gb_core / n_perfect

    print(f"\nBest GB core atom count : {best_n_gb_core}")
    print(f"Best normalized density : {best_normalized:.4f}")

    # --- Save outputs ---
    log_path = output_dir / "density_gbe_log.json"
    with open(log_path, "w") as f:
        json.dump(density_gbe_log, f, separators=(",", ":"))
    print(f"Saved {log_path}  ({len(density_gbe_log)} evaluations)")

    if best_dump and Path(best_dump).exists():
        dest = output_dir / "best_density_structure.dump"
        shutil.copy(best_dump, dest)
        print(f"Saved {dest}")
    else:
        print("WARNING: best dump file not found; best_density_structure.dump not saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Density optimization for a Σ5[001]{310} GB")
    parser.add_argument(
        "--material", required=True, choices=list(MATERIAL_PARAMS),
        help="Material to optimize (e.g. Fe, Ni)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent   # density_optimization/
    output_dir = script_dir / args.material / BOUNDARY
    work_dir = output_dir
    lammps_dir = script_dir / "lammps"
    input_script = str(lammps_dir / "lmp.in")

    # lmp.in uses `include ../../potential.setup` and `include ../../min_strat.commands`,
    # which resolve to work_dir/ when LAMMPS runs from workdir.X/gen_Y/.
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(lammps_dir / args.material / "potential.setup", work_dir / "potential.setup")
    shutil.copy(lammps_dir / "min_strat.commands", work_dir / "min_strat.commands")

    params = MATERIAL_PARAMS[args.material]
    run_density_optimization(
        **params,
        output_dir=output_dir,
        work_dir=work_dir,
        repeat_factor=(5, 9),
        lmp_binary="lmp",
        input_script=input_script,
        job_walltime="06:00:00",
        population_size=50,
        generations=200,
        seed=0,
    )


if __name__ == "__main__":
    main()
