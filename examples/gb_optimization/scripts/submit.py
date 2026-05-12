#! /usr/bin/env python
import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MATERIALS_DIR = PROJECT_ROOT / "materials"


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a GBOpt job to SLURM")
    parser.add_argument("--material", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--method", required=True, choices=["GA", "MC"])
    args = parser.parse_args()

    method_key = args.method.lower()
    with open(CONFIG_DIR / f"{method_key}.toml", "rb") as f:
        method_cfg = tomllib.load(f)

    with open(MATERIALS_DIR / f"{args.material}.toml", "rb") as f:
        tomllib.load(f)  # validate file exists and is valid TOML

    slurm = method_cfg.get("slurm", {})
    job_name = f"{args.material}_{args.boundary}_{args.method}"
    script_path = SCRIPTS_DIR / f"run_{method_key}.py"

    lines = [
        "#!/bin/bash",
        f'#SBATCH -J "{job_name}"',
        f"#SBATCH -o {job_name}_%j.out",
        f"#SBATCH -e {job_name}_%j.err",
    ]
    if "time" in slurm:
        lines.append(f"#SBATCH --time={slurm['time']}")
    if "ntasks_per_node" in slurm:
        lines.append(f"#SBATCH --ntasks-per-node={slurm['ntasks_per_node']}")
    if "cpus_per_task" in slurm:
        lines.append(f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}")
    if "nodes" in slurm:
        lines.append(f"#SBATCH --nodes={slurm['nodes']}")
    if "wckey" in slurm:
        lines.append(f"#SBATCH --wckey {slurm['wckey']}")
    if "partition" in slurm:
        lines.append(f"#SBATCH -p {slurm['partition']}")
    lines.append("")
    if "module" in slurm:
        lines.append(f"ml {slurm['module']}")
    lines.append("source ~/miniforge/etc/profile.d/conda.sh")
    lines.append("conda activate GBOpt")
    if "omp_num_threads" in slurm:
        lines.append(f"export OMP_NUM_THREADS={slurm['omp_num_threads']}")
    lines += [
        "",
        f"python {script_path} --material {args.material} --boundary {args.boundary}",
        "",
    ]

    slurm_file = Path(f"{job_name}.slurm")
    slurm_file.write_text("\n".join(lines))
    print(f"Written {slurm_file}")

    result = subprocess.run(["sbatch", str(slurm_file)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
