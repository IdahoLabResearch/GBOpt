# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import functools
import json
import pickle
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SCHEMA_VERSION: int = 1
"""Schema version written into every checkpoint envelope."""


class CheckpointError(Exception):
    """Base exception for the Checkpoint module."""


class CheckpointValueError(CheckpointError, ValueError):
    """Raised when an argument has an invalid value."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when persisted checkpoint identity does not match the current run."""


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and Paths to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _validate_checkpoint_format(fmt: str) -> None:
    """Validate a supported checkpoint serialization format.

    :param fmt: Serialization format name.
    :raises CheckpointValueError: If ``fmt`` is unsupported.
    """
    if fmt not in ("json", "pickle"):
        raise CheckpointValueError(
            f"fmt must be 'json' or 'pickle', got {fmt!r}"
        )


def _load_checkpoint_payload(path: Path, fmt: str, *, kind: str) -> Any:
    """Deserialize one checkpoint payload with consistent error translation.

    :param path: Checkpoint file path.
    :param fmt: Serialization format.
    :param kind: Keyword argument, required. Human-readable checkpoint kind used in
        diagnostics.
    :return: Deserialized payload.
    :raises CheckpointError: If the file cannot be parsed.
    """
    try:
        if fmt == "json":
            with open(path) as fp:
                return json.load(fp)
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception as exc:
        raise CheckpointError(
            f"Could not parse {kind} file {path}: {exc}"
        ) from exc


def _save_checkpoint_payload(
    path: Path,
    fmt: str,
    payload: Any,
    *,
    kind: str,
    json_indent: int | None = None,
) -> None:
    """Atomically serialize one checkpoint payload via a temporary file.

    :param path: Checkpoint file path.
    :param fmt: Serialization format.
    :param payload: State to serialize.
    :param kind: Keyword argument, required. Human-readable checkpoint kind used in
        diagnostics.
    :param json_indent: Keyword argument, optional, defaults to ``None``. JSON
        indentation level.
    :raises CheckpointError: If the payload cannot be written.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if fmt == "json":
            with open(tmp, "w") as fp:
                json.dump(_to_serializable(payload), fp, indent=json_indent)
        else:
            with open(tmp, "wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(str(tmp), str(path))
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise CheckpointError(
            f"Failed to save {kind} to {path}: {exc}"
        ) from exc


def _delete_checkpoint_file(path: Path) -> None:
    """Delete one checkpoint file when it exists.

    :param path: Checkpoint file path.
    """
    if path.exists():
        path.unlink()


class CheckpointStore:
    """Centralizes checkpoint persistence for a minimizer run.

    This class implements the *null-object pattern*: :meth:`disabled` returns an
    instance whose methods are all no-ops, so callers never need ``if checkpoint is not
    None:`` guards.

    Typical usage in a minimizer loop::

        checkpoint = CheckpointStore.from_optional(
            checkpoint_file, fmt=checkpoint_format, interval=checkpoint_interval
        )
        # ... loop ...
        checkpoint.save_if_due(step, lambda: build_state(step))
        # On convergence:
        checkpoint.save_final(build_state(step))

    Obtain an instance via :meth:`from_optional` or :meth:`disabled`.
    """

    def __init__(
        self,
        path: Path,
        fmt: str,
        interval: int,
    ):
        self._path = path
        self._fmt = fmt
        self._interval = interval
        self._disabled = False

    @classmethod
    def disabled(cls) -> CheckpointStore:
        """Return a no-op store; all methods are safe to call and do nothing.

        This is returned automatically by :meth:`from_optional` when *path* is ``None``.
        You rarely need to call this directly::

            store = CheckpointStore.disabled()
            store.save_if_due(1, lambda: {...}) # silently ignored
            assert store.load() is None
        """
        store = cls.__new__(cls)
        store._disabled = True
        return store

    @classmethod
    def from_optional(
        cls,
        path: Path | str | None,
        fmt: str = "json",
        interval: int = 1,
    ) -> CheckpointStore:
        """Return :meth:`disabled` when *path* is ``None``, else a live store.

        :param path: Destination path, or ``None`` to disable checkpointing.
        :param fmt: Serialization format — ``"json"`` (default) or ``"pickle"``.
        :param interval: Persist state every *interval* steps/generations.
        :raises CheckpointValueError: If *fmt* is not ``"json"`` or ``"pickle"``, or if
            *interval* is less than 1

        Example::

            # Checkpointing enabled - saves every 5 steps:
            store = CheckpointStore.from_optional("run.json", interval=5)

            # Checkpointing disabled - all calls are no-ops:
            store = CheckpointStore.from_optional(None)
            assert not store.enabled
        """
        if path is None:
            return cls.disabled()
        _validate_checkpoint_format(fmt)
        if interval < 1:
            raise CheckpointValueError(f"interval must be >= 1, got {interval!r}")
        return cls(Path(path), fmt, interval)

    @property
    def enabled(self) -> bool:
        """``True`` if this store will actually persist state."""
        return not self._disabled

    @property
    def exists(self) -> bool:
        """``True`` if the checkpoint file is present on disk."""
        return not self._disabled and self._path.exists()

    def load(self) -> dict | None:
        """Return the deserialized state dict, or ``None`` if no file exists.

        :return: State dict, or ``None`` when no checkpoint is present.
        :raises CheckpointError: If the file exists but cannot be parsed.
        """
        if self._disabled or not self._path.exists():
            return None
        return _load_checkpoint_payload(
            self._path,
            self._fmt,
            kind="checkpoint",
        )

    def is_due(self, index: int) -> bool:
        """Return ``True`` when :meth:`save_if_due` at *index* would actually write
        state.
        """
        return not self._disabled and index % self._interval == 0

    def save_if_due(self, index: int, state_fn: Callable[[], dict]) -> None:
        """Persist state when *index* is a multiple of the configured interval.

        :param index: Current step or generation index.
        :param state_fn: Zero-argument callable returning the state dict. Called only
            when a save is actually due.
        """
        if self._disabled:
            return
        if index % self._interval == 0:
            self._save(state_fn())

    def save_final(self, state: dict) -> None:
        """Unconditionally persist *state*, bypassing the interval check.

        Use for convergence or early-termination saves. No-op when disabled.

        :param state: State dict to persist.
        """
        if self._disabled:
            return
        self._save(state)

    def delete(self) -> None:
        """Remove the checkpoint file if it exists.  No-op when disabled."""
        if self._disabled:
            return
        _delete_checkpoint_file(self._path)

    def _save(self, state: dict) -> None:
        """Atomically write *state* to disk via a tmp-then-move."""
        _save_checkpoint_payload(
            self._path,
            self._fmt,
            state,
            kind="checkpoint",
            json_indent=2,
        )


class CandidateCheckpoint:
    """Per-candidate result cache for a single optimization iteration.

    Manages its own checkpoint file (sibling to the main run checkpoint, named
    ``{main_stem}.iter{N}{ext}``). Each call to :meth:`record` atomically persists state
    to disk, so a crash between candidate evaluations leaves a recoverable snapshot.

    This class is used internally by
    :class:`~GBOpt.GBMinimizer.GeneticAlgorithmMinimizer`. It is not normally
    instantiated directly by users; the minimizer creates and manages it for each
    iteration.
    """

    def __init__(self, path: Path, fmt: str, iteration_index: int, unique_ids: list):
        """
        :param path: Path to this iteration's checkpoint file.
        :param fmt: Serialization format — ``"json"`` or ``"pickle"``.
        :param iteration_index: Zero-based iteration index this checkpoint covers.
        :param unique_ids: Ordered list of candidate unique IDs for this iteration.
        :raises CheckpointValueError: If *fmt* is not ``"json"`` or ``"pickle"``.
        """
        _validate_checkpoint_format(fmt)
        if not isinstance(unique_ids, list) or not all(
            isinstance(unique_id, str) and unique_id for unique_id in unique_ids
        ):
            raise CheckpointValueError(
                "unique_ids must be a list of non-empty strings"
            )
        if len(set(unique_ids)) != len(unique_ids):
            raise CheckpointValueError("unique_ids must not contain duplicates")
        self._path = path
        self._fmt = fmt
        self.iteration_index = iteration_index
        self._unique_ids = tuple(unique_ids)
        # None → not yet evaluated; dict → result payload. ``metadata`` is optional so
        # legacy checkpoints remain readable while richer optimizer paths can retain
        # typed failure context without serializing live objects.
        self._results: dict = {uid: None for uid in self._unique_ids}

    # factories

    @classmethod
    def new_or_resume(
        cls,
        main_path: Path,
        fmt: str,
        iteration_index: int,
        unique_ids: list,
    ) -> CandidateCheckpoint:
        """Return a checkpoint loaded from disk if the iter file exists, else a fresh
        one.

        :param main_path: Path to the parent run's main checkpoint file.
        :param fmt: Serialization format — ``"json"`` or ``"pickle"``.
        :param iteration_index: Zero-based iteration index.
        :param unique_ids: Ordered list of candidate unique IDs.
        :return: A :class:`CandidateCheckpoint` instance.
        :raises CheckpointCompatibilityError: If an existing sidecar does not match the
            requested iteration and ordered candidate population.
        :raises CheckpointError: If an existing sidecar cannot be parsed.
        :raises CheckpointValueError: If checkpoint configuration is invalid.
        """
        iter_path = cls._derive_path(main_path, iteration_index)
        if iter_path.exists():
            return cls._load(iter_path, fmt, iteration_index, unique_ids)
        return cls(iter_path, fmt, iteration_index, unique_ids)

    @staticmethod
    def _derive_path(main_path: Path, iteration_index: int) -> Path:
        """Compute the sidecar file path for *iteration_index* next to *main_path*."""
        return main_path.with_suffix(f".iter{iteration_index}{main_path.suffix}")

    @classmethod
    def _load(
        cls,
        path: Path,
        fmt: str,
        iteration_index: int,
        unique_ids: list,
    ) -> CandidateCheckpoint:
        """Restore a :class:`CandidateCheckpoint` from an existing file.

        :param path: Path to the iter checkpoint file.
        :param fmt: Serialization format — ``"json"`` or ``"pickle"``.
        :param iteration_index: Zero-based iteration index.
        :param unique_ids: Ordered list of candidate unique IDs expected for this
            iteration.
        :return: A populated :class:`CandidateCheckpoint`.
        :raises CheckpointError: If the file cannot be parsed.
        :raises CheckpointCompatibilityError: If the persisted iteration or candidate
            population identity does not exactly match the requested state.
        """
        payload = _load_checkpoint_payload(
            path,
            fmt,
            kind="candidate checkpoint",
        )

        if not isinstance(payload, dict):
            raise CheckpointCompatibilityError(
                f"Candidate checkpoint {path} does not contain a mapping payload"
            )

        saved_iteration = payload.get("iteration_index")
        if saved_iteration != iteration_index:
            raise CheckpointCompatibilityError(
                f"Candidate checkpoint iteration mismatch: saved {saved_iteration!r}, "
                f"requested {iteration_index!r}"
            )

        saved_results = payload.get("results")
        if not isinstance(saved_results, dict):
            raise CheckpointCompatibilityError(
                f"Candidate checkpoint {path} does not contain a valid results mapping"
            )

        # Older candidate sidecars predate the explicit ``unique_ids`` field, but they
        # persisted every candidate key (including unfinished ``None`` entries). Their
        # ordered results keys therefore provide an equivalent strict population
        # identity for compatibility validation.
        saved_unique_ids = payload.get("unique_ids", list(saved_results))
        if not isinstance(saved_unique_ids, list) or not all(
            isinstance(unique_id, str) for unique_id in saved_unique_ids
        ):
            raise CheckpointCompatibilityError(
                f"Candidate checkpoint {path} has invalid population identity metadata"
            )

        expected_unique_ids = tuple(unique_ids)
        persisted_unique_ids = tuple(saved_unique_ids)
        if persisted_unique_ids != expected_unique_ids:
            raise CheckpointCompatibilityError(
                "Candidate checkpoint population identity does not match the current "
                "ordered candidate population"
            )
        if tuple(saved_results) != persisted_unique_ids:
            raise CheckpointCompatibilityError(
                "Candidate checkpoint result keys do not match its population identity"
            )

        obj = cls(path, fmt, iteration_index, unique_ids)
        for uid in obj._unique_ids:
            obj._results[uid] = saved_results[uid]
        return obj

    # query / record

    def is_done(self, unique_id: str) -> bool:
        """Return ``True`` if this candidate has already been evaluated (via
        `unique_id`).
        """
        return self._results.get(unique_id) is not None

    def get_result(self, unique_id: str) -> tuple:
        """Return ``(energy, dump_path)`` for an already-evaluated candidate.

        :param unique_id: The candidate's unique ID.
        :return: Tuple of ``(grain_boundary_energy, dump_file_path)``. *dump_file_path*
            is ``None`` when the evaluation failed.
        :raises CheckpointError: If the candidate has not been evaluated yet.
        """
        r = self._results.get(unique_id)
        if r is None:
            raise CheckpointError(
                f"No result recorded for {unique_id!r}. "
                "Check is_done() before calling get_result()."
            )
        return (r["energy"], r["dump"])

    def get_metadata(self, unique_id: str) -> dict | None:
        """Return optional optimizer-specific metadata for a completed candidate.

        :param unique_id: The candidate's unique ID.
        :return: A copied metadata dictionary, or ``None`` for a legacy result.
        :raises CheckpointError: If the candidate has not been evaluated yet.
        """
        result = self._results.get(unique_id)
        if result is None:
            raise CheckpointError(
                f"No result recorded for {unique_id!r}. Check is_done() before calling "
                "get_metadata()."
            )
        metadata = result.get("metadata")
        if metadata is None:
            return None
        return dict(metadata)

    def record(
        self,
        unique_id: str,
        energy: float,
        dump: str | None,
        *,
        metadata: dict | None = None,
    ) -> None:
        """Record a candidate result and atomically persist the checkpoint to disk.

        :param unique_id: The candidate's unique ID.
        :param energy: Evaluated grain boundary energy.
        :param dump: Path to the output dump file, or ``None`` on failure.
        :param metadata: Keyword argument, optional, defaults to ``None``. JSON-safe
            optimizer-specific result metadata.
        :raises CheckpointValueError: If ``unique_id`` is not part of this checkpoint's
            candidate population.
        :raises CheckpointError: If the file cannot be written.
        """
        if unique_id not in self._results:
            raise CheckpointValueError(
                f"Candidate {unique_id!r} is not part of this checkpoint population"
            )
        payload = {"energy": float(energy), "dump": dump}
        if metadata is not None:
            payload["metadata"] = metadata
        self._results[unique_id] = payload
        self._save()

    def delete(self) -> None:
        """Remove the checkpoint file from disk if it exists."""
        _delete_checkpoint_file(self._path)

    # persistence

    def _save(self) -> None:
        """Atomically write checkpoint state to disk via a tmp-then-move."""
        payload = {
            "iteration_index": self.iteration_index,
            "unique_ids": list(self._unique_ids),
            "results": self._results,
        }
        _save_checkpoint_payload(
            self._path,
            self._fmt,
            payload,
            kind="candidate checkpoint",
        )


def _wrap_batch_func_with_checkpoint(
    batch_func: Callable,
    *,
    penalty: float,
) -> Callable:
    """Wrap a batch energy function that lacks a ``checkpoint`` kwarg.

    The wrapper accepts and passes a :class:`CandidateCheckpoint` but records results
    only *after* the underlying batch call returns (batch-level granularity). For
    per-job recovery, the batch function should declare ``checkpoint=None`` in its own
    signature and call :meth:`CandidateCheckpoint.record` as each individual job
    completes.

    :param batch_func: Original batch energy function to wrap.
    :param penalty: Keyword argument, required. Optimizer-owned failure energy used when
        a returned result omits ``energy``.
    :return: A wrapped callable that accepts a ``checkpoint`` keyword argument.
    """

    @functools.wraps(batch_func)
    def _wrapped(GB, manips, structs, lineages, unique_ids, checkpoint=None):
        results = batch_func(GB, manips, structs, lineages, unique_ids)
        if checkpoint is not None:
            for uid, result in zip(unique_ids, results):
                if not checkpoint.is_done(uid):
                    checkpoint.record(
                        uid,
                        float(result.get("energy", penalty)),
                        result.get("final_dump", None),
                    )
        return results

    return _wrapped
