# FILE: GBOpt/slurm_utils.py

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional


SlurmState = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]


@dataclass
class SlurmJob:
    job_id: int
    script_path: Path


class SlurmError(RuntimeError):
    pass


def submit_job(script_path: Path, *, extra_sbatch_args: Optional[list[str]] = None) -> SlurmJob:
    """
    Submit a SLURM batch script and return a SlurmJob.

    script_path
        Path to a file containing #!/bin/bash and #SBATCH lines.

    extra_sbatch_args
        Optional extra arguments for sbatch (e.g. ["--dependency=afterok:12345"]).
    """
    if extra_sbatch_args is None:
        extra_sbatch_args = []

    cmd = ["sbatch", *extra_sbatch_args, str(script_path)]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Typical output: "Submitted batch job 12345\n"
    line = res.stdout.strip()
    parts = line.split()
    try:
        job_id = int(parts[-1])
    except (ValueError, IndexError) as e:
        raise SlurmError(f"Could not parse job id from sbatch output: {line!r}") from e

    return SlurmJob(job_id=job_id, script_path=Path(script_path))


def get_job_state(job_id: int) -> SlurmState:
    """
    Return SLURM job state for job_id.

    Tries sacct first (works for finished jobs),
    falls back to squeue (for running/pending).
    """
    # Try sacct (covers completed / failed / cancelled)
    sacct_cmd = [
        "sacct",
        "-j", str(job_id),
        "--format=JobIDRaw,State",
        "--noheader",
        "--parsable2",
    ]
    res = subprocess.run(sacct_cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    if out:
        # Format like: "12345|COMPLETED"
        first_line = out.splitlines()[0]
        fields = first_line.split("|")
        if len(fields) >= 2:
            state = fields[1].strip().upper()
            return _normalize_state(state)

    # Fallback: squeue (only sees active jobs)
    squeue_cmd = [
        "squeue",
        "-j", str(job_id),
        "--noheader",
        "--format=%T",
    ]
    res = subprocess.run(squeue_cmd, capture_output=True, text=True)
    state = res.stdout.strip()
    if not state:
        # Not in queue, sacct gave nothing either:
        return "UNKNOWN"
    return _normalize_state(state)


def _normalize_state(raw: str) -> SlurmState:
    raw = raw.upper()
    if "PEND" in raw:
        return "PENDING"
    if "RUN" in raw:
        return "RUNNING"
    if "COMP" in raw:
        return "COMPLETED"
    if "CANCEL" in raw or "TIMEOUT" in raw:
        return "CANCELLED"
    if "FAIL" in raw or "NODE_FAIL" in raw:
        return "FAILED"
    return "UNKNOWN"


def wait_for_jobs(
    jobs: Iterable[SlurmJob] | Iterable[int],
    *,
    poll_interval: int = 10,
    fail_on: tuple[SlurmState, ...] = ("FAILED", "CANCELLED"),
    timeout: Optional[int] = None,
) -> None:
    """
    Block until all jobs finish (COMPLETED/FAILED/CANCELLED/UNKNOWN).

    jobs
        Iterable of SlurmJob or plain job_ids.

    poll_interval
        Seconds between queries.

    fail_on
        If any job ends in one of these states, raise SlurmError.

    timeout
        Optional wall-clock timeout in seconds for the wait loop.

    Raises
    ------
    SlurmError if any job ends in a 'fail_on' state or timeout is hit.
    """
    # Normalize to job ids
    job_ids: List[int] = []
    for j in jobs:
        if isinstance(j, SlurmJob):
            job_ids.append(j.job_id)
        else:
            job_ids.append(int(j))

    remaining = set(job_ids)
    start = time.time()

    while remaining:
        for jid in list(remaining):
            state = get_job_state(jid)
            if state in ("COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"):
                remaining.remove(jid)
                if state in fail_on:
                    raise SlurmError(f"Job {jid} ended with state {state}")
        if not remaining:
            break

        if timeout is not None and (time.time() - start) > timeout:
            raise SlurmError(f"Timed out waiting for jobs: {sorted(remaining)}")

        time.sleep(poll_interval)
