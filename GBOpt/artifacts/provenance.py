# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Persist artifact lifecycle provenance without becoming restart state.

This module consumes immutable artifact-store snapshots and lifecycle notifications and
writes two lightweight products beneath a run-owned artifact root: an atomically
replaced ``manifest.json`` describing current artifact state and an append-only
``history.jsonl`` describing lifecycle events. It does not decide retention membership,
perform optimizer selection, serialize checkpoints, or remove filesystem artifacts.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path

import numpy as np

from GBOpt.artifacts._paths import _normalize_path
from GBOpt.artifacts.types import ArtifactError, ArtifactRecord, RetentionCandidate

_MANIFEST_VERSION = 2
_HISTORY_VERSION = 1


class ArtifactProvenanceError(ArtifactError):
    """Raised when artifact provenance state cannot be validated or persisted."""


def _require_nonempty_string(value: object, *, name: str) -> str:
    """Validate one non-empty provenance string field.

    :param value: Candidate string value.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated string.
    :raises ArtifactProvenanceError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ArtifactProvenanceError(f"{name} must be a non-empty string")
    return value


def _canonical_json(value: object, *, context: str) -> str:
    """Return deterministic compact JSON for one provenance value.

    :param value: JSON-compatible value.
    :param context: Keyword argument, required. Human-readable diagnostic context.
    :return: Deterministically ordered JSON text.
    :raises ArtifactProvenanceError: If ``value`` is not JSON serializable.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactProvenanceError(
            f"{context} must contain only JSON-serializable finite values"
        ) from exc


def _json_safe_value(value: object, *, context: str) -> object:
    """Normalize supported provenance metadata to detached plain JSON values.

    Checkpoint-owned reconstruction metadata contains NumPy arrays and scalar values.
    Provenance serializes equivalent data without importing checkpoint serialization
    helpers, preserving the architectural separation between restart and observability.

    :param value: Candidate metadata value.
    :param context: Keyword argument, required. Human-readable diagnostic context.
    :return: Plain JSON-compatible value with NumPy/Path values normalized.
    :raises ArtifactProvenanceError: If ``value`` contains an unsupported type, a
        non-string mapping key, or a non-finite floating-point value.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ArtifactProvenanceError(f"{context} contains a non-finite value")
        return normalized
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [
            _json_safe_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value.tolist())
        ]
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise ArtifactProvenanceError(
                f"{context} contains a non-string mapping key"
            )
        result: dict[str, object] = {}
        for key in sorted(keys):
            result[key] = _json_safe_value(value[key], context=f"{context}.{key}")
        return result
    raise ArtifactProvenanceError(
        f"{context} contains unsupported type {type(value).__name__}"
    )


def _json_copy(value: object, *, context: str) -> object:
    """Return a detached plain-JSON copy of one supported metadata value.

    :param value: Provenance metadata value.
    :param context: Keyword argument, required. Human-readable diagnostic context.
    :return: Detached JSON-compatible copy.
    :raises ArtifactProvenanceError: If ``value`` cannot be normalized safely.
    """
    normalized = _json_safe_value(value, context=context)
    encoded = _canonical_json(normalized, context=context)
    return json.loads(encoded)


def _normalize_calculation_context(
    value: object,
) -> dict[str, object] | None:
    """Validate and detach optional run-level calculation provenance.

    :param value: Mapping supplied by the evaluator/campaign, or ``None``.
    :return: Deterministically normalized plain-JSON mapping, or ``None``.
    :raises ArtifactProvenanceError: If ``value`` is not a mapping or contains
        unsupported provenance values.
    """
    normalized: dict[str, object] | None = None
    if value is not None:
        if not isinstance(value, Mapping):
            raise ArtifactProvenanceError(
                "calculation_context must be a mapping or None"
            )
        copied = _json_copy(value, context="calculation_context")
        if not isinstance(copied, dict):
            raise ArtifactProvenanceError(
                "calculation_context must normalize to a JSON object"
            )
        normalized = copied
    return normalized


def _require_nonnegative_int(value: object, *, name: str) -> int:
    """Validate one non-negative provenance integer.

    :param value: Candidate integer value.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated Python integer.
    :raises ArtifactProvenanceError: If ``value`` is Boolean, non-integral, or negative.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ArtifactProvenanceError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ArtifactProvenanceError(f"{name} must be a non-negative integer")
    return normalized


class _ArtifactProvenance:
    """Write current artifact manifests and append-only lifecycle history.

    Existing history entries are loaded as canonical JSON so replay after a crash does
    not duplicate an identical lifecycle event. Provenance files are intentionally not
    consulted for optimizer restart; callers reconstruct authoritative state from normal
    checkpoints and then regenerate the manifest from that state.

    :param root: Run-owned artifact directory containing provenance files.
    :param calculation_context: Keyword argument, optional, defaults to ``None``.
        Run-level evaluator/campaign provenance shared by retained/final results.
    :raises ArtifactProvenanceError: If configuration is invalid or an existing history
        file is malformed.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        calculation_context: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize one provenance writer for a run-owned artifact root.

        :param root: Run-owned artifact directory containing provenance files.
        :param calculation_context: Keyword argument, optional, defaults to ``None``.
            Run-level evaluator/campaign provenance shared by retained/final results.
        :raises ArtifactProvenanceError: If ``root`` or ``calculation_context`` is
            invalid, or an existing history file is malformed.
        """
        self._root = _normalize_path(
            root, name="root", error_type=ArtifactProvenanceError
        )
        self._manifest_path = self._root / "manifest.json"
        self._history_path = self._root / "history.jsonl"
        self._calculation_context = _normalize_calculation_context(calculation_context)
        self._validate_existing_calculation_context()
        self._history_entries = self._load_existing_history()

    def _validate_existing_calculation_context(self) -> None:
        """Reject silent replacement of run-level calculation provenance.

        Existing provenance is not restart-authoritative, so a mismatch does not
        invalidate optimizer state. Initialization fails instead, which causes callers
        to retain source artifacts rather than overwrite the established run context.

        :raises ArtifactProvenanceError: If an existing manifest is malformed or its
            calculation context differs from the configured context.
        """
        if not self._manifest_path.exists():
            return
        try:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactProvenanceError(
                f"existing artifact manifest could not be read: {self._manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise ArtifactProvenanceError(
                "existing artifact manifest must contain a JSON object"
            )
        if manifest.get("calculation_context") != self._calculation_context:
            raise ArtifactProvenanceError(
                "existing artifact calculation_context does not match the configured "
                "run context"
            )

    @property
    def manifest_path(self) -> Path:
        """Return the current-state manifest path."""
        return self._manifest_path

    @property
    def history_path(self) -> Path:
        """Return the append-only lifecycle history path."""
        return self._history_path

    def _load_existing_history(self) -> set[str]:
        """Load canonical existing history entries for idempotent replay.

        :return: Canonical JSON strings already present in the history file.
        :raises ArtifactProvenanceError: If the history file cannot be read or contains
            a malformed/non-object JSONL entry.
        """
        if not self._history_path.exists():
            return set()
        entries: set[str] = set()
        try:
            with self._history_path.open("r", encoding="utf-8") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.endswith("\n"):
                        raise ArtifactProvenanceError(
                            "artifact history contains an incomplete trailing entry"
                        )
                    stripped = raw_line.strip()
                    if not stripped:
                        raise ArtifactProvenanceError(
                            f"artifact history line {line_number} is empty"
                        )
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ArtifactProvenanceError(
                            f"artifact history line {line_number} is invalid JSON"
                        ) from exc
                    if not isinstance(event, dict):
                        raise ArtifactProvenanceError(
                            f"artifact history line {line_number} must be a JSON object"
                        )
                    entries.add(
                        _canonical_json(event, context="artifact history entry")
                    )
        except OSError as exc:
            raise ArtifactProvenanceError(
                f"could not read artifact history {self._history_path}"
            ) from exc
        return entries

    def _append_event(self, event: dict[str, object]) -> None:
        """Append one deterministic lifecycle event unless it already exists.

        :param event: Complete JSON-safe event object.
        :raises ArtifactProvenanceError: If the event is malformed or cannot be
            persisted.
        """
        event = {"version": _HISTORY_VERSION, **event}
        encoded = _canonical_json(event, context="artifact history event")
        if encoded in self._history_entries:
            return
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with self._history_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ArtifactProvenanceError(
                f"could not append artifact history {self._history_path}"
            ) from exc
        self._history_entries.add(encoded)

    def _record_retention_reason_event(
        self,
        event: str,
        candidate_id: str,
        reason: str,
    ) -> None:
        """Record one validated retention-reason lifecycle event.

        :param event: Lifecycle event name.
        :param candidate_id: Stable logical candidate identity.
        :param reason: Scientific retention reason.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._append_event(
            {
                "event": event,
                "candidate_id": _require_nonempty_string(
                    candidate_id, name="candidate_id"
                ),
                "reason": _require_nonempty_string(reason, name="reason"),
            }
        )

    def _record_path_event(
        self,
        event: str,
        candidate_id: str,
        path: str | os.PathLike[str],
    ) -> None:
        """Record one validated candidate/path lifecycle event.

        :param event: Lifecycle event name.
        :param candidate_id: Stable logical candidate identity.
        :param path: Filesystem path associated with the event.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._append_event(
            {
                "event": event,
                "candidate_id": _require_nonempty_string(
                    candidate_id, name="candidate_id"
                ),
                "path": str(
                    _normalize_path(
                        path, name="path", error_type=ArtifactProvenanceError
                    )
                ),
            }
        )

    def record_candidate_evaluated(self, candidate: RetentionCandidate) -> None:
        """Record one successful validated candidate evaluation.

        :param candidate: Candidate that reached scientific retention classification.
        :raises ArtifactProvenanceError: If ``candidate`` is invalid or history cannot
            be persisted.
        """
        if not isinstance(candidate, RetentionCandidate):
            raise ArtifactProvenanceError("candidate must be a RetentionCandidate")
        self._append_event(
            {
                "event": "candidate_evaluated",
                "candidate_id": candidate.candidate_id,
                "generation": candidate.generation,
                "objective": candidate.objective,
                "evaluation_status": "success",
                "lineage": list(candidate.lineage),
            }
        )

    def record_evaluation_failed(
        self,
        candidate_id: str,
        generation: int,
        failure_reason: str,
        *,
        diagnostic_path: str | os.PathLike[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Record one failed evaluation without promoting it to retention state.

        :param candidate_id: Stable logical candidate identity.
        :param generation: Optimizer generation/step where evaluation occurred.
        :param failure_reason: Human-readable durable failure context.
        :param diagnostic_path: Keyword argument, optional, defaults to ``None``.
            Evaluator artifact path available for bounded diagnostic retention.
        :param metadata: Keyword argument, optional, defaults to ``None``. Additional
            JSON-safe evaluator/campaign diagnostic identifiers.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        normalized_path = None
        if diagnostic_path is not None:
            normalized_path = str(
                _normalize_path(
                    diagnostic_path,
                    name="diagnostic_path",
                    error_type=ArtifactProvenanceError,
                )
            )
        normalized_metadata = None
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ArtifactProvenanceError("metadata must be a mapping or None")
            normalized_metadata = _json_copy(
                metadata, context="failed evaluation metadata"
            )
        self._append_event(
            {
                "event": "evaluation_failed",
                "candidate_id": _require_nonempty_string(
                    candidate_id, name="candidate_id"
                ),
                "generation": _require_nonnegative_int(generation, name="generation"),
                "evaluation_status": "failure",
                "failure_reason": _require_nonempty_string(
                    failure_reason, name="failure_reason"
                ),
                "diagnostic_path": normalized_path,
                "metadata": normalized_metadata,
            }
        )

    def record_failure_diagnostic_pruned(
        self,
        candidate_id: str,
        path: str | os.PathLike[str],
    ) -> None:
        """Record removal of one bounded failed-evaluation diagnostic source.

        :param candidate_id: Stable logical candidate identity.
        :param path: Evaluator diagnostic source removed after a durable checkpoint.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._record_path_event("failure_diagnostic_pruned", candidate_id, path)

    def record_properties_calculated(self, candidate: RetentionCandidate) -> None:
        """Record the normalized property mapping calculated for one candidate.

        :param candidate: Candidate whose retention properties were validated.
        :raises ArtifactProvenanceError: If ``candidate`` is invalid or history cannot
            be persisted.
        """
        if not isinstance(candidate, RetentionCandidate):
            raise ArtifactProvenanceError("candidate must be a RetentionCandidate")
        candidate_state = candidate.to_state()
        self._append_event(
            {
                "event": "properties_calculated",
                "candidate_id": candidate.candidate_id,
                "generation": candidate.generation,
                "properties": candidate_state["properties"],
            }
        )

    def record_retention_reason_added(self, candidate_id: str, reason: str) -> None:
        """Record one scientific retention reason becoming active.

        :param candidate_id: Stable logical candidate identity.
        :param reason: Scientific retention reason, normally ``rule:<name>``.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._record_retention_reason_event(
            "retention_reason_added", candidate_id, reason
        )

    def record_retention_reason_removed(self, candidate_id: str, reason: str) -> None:
        """Record one scientific retention reason becoming inactive.

        :param candidate_id: Stable logical candidate identity.
        :param reason: Scientific retention reason, normally ``rule:<name>``.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._record_retention_reason_event(
            "retention_reason_removed", candidate_id, reason
        )

    def record_archive_created(
        self, candidate_id: str, path: str | os.PathLike[str]
    ) -> None:
        """Record creation or refresh of one canonical retained structure.

        :param candidate_id: Stable logical candidate identity.
        :param path: Canonical retained structure path.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._record_path_event("archive_created", candidate_id, path)

    def record_source_pruned(
        self, candidate_id: str, path: str | os.PathLike[str]
    ) -> None:
        """Record successful cleanup of one evaluator-owned source artifact.

        :param candidate_id: Stable logical candidate identity.
        :param path: Evaluator-returned source structure path used for cleanup dispatch.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._record_path_event("source_pruned", candidate_id, path)

    def record_archive_evicted(
        self, candidate_id: str, path: str | os.PathLike[str]
    ) -> None:
        """Record successful deletion of one no-longer-retained canonical structure.

        :param candidate_id: Stable logical candidate identity.
        :param path: Canonical archive path removed after checkpoint commit.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        self._record_path_event("archive_evicted", candidate_id, path)

    def record_cleanup_failed(
        self,
        operation: str,
        path: str | os.PathLike[str],
        message: str,
        *,
        candidate_id: str | None = None,
    ) -> None:
        """Record one best-effort cleanup failure without changing optimizer state.

        :param operation: Lifecycle operation that failed, for example ``source_prune``
            or ``archive_evict``.
        :param path: Cleanup target or evaluator source path associated with the
            failure.
        :param message: Failure diagnostic text.
        :param candidate_id: Keyword argument, optional, defaults to ``None``. Candidate
            identity when the cleanup target is candidate-specific.
        :raises ArtifactProvenanceError: If fields are invalid or history cannot be
            persisted.
        """
        normalized_candidate_id = None
        if candidate_id is not None:
            normalized_candidate_id = _require_nonempty_string(
                candidate_id, name="candidate_id"
            )
        self._append_event(
            {
                "event": "cleanup_failed",
                "operation": _require_nonempty_string(operation, name="operation"),
                "candidate_id": normalized_candidate_id,
                "path": str(
                    _normalize_path(
                        path, name="path", error_type=ArtifactProvenanceError
                    )
                ),
                "message": _require_nonempty_string(message, name="message"),
            }
        )

    def write_manifest(
        self,
        records: Sequence[ArtifactRecord],
        *,
        ownership_metadata: Mapping[str, object] | None = None,
        failure_diagnostics: Sequence[Mapping[str, object]] = (),
    ) -> None:
        """Atomically replace the current artifact manifest from authoritative
        snapshots.

        Candidate records are sorted lexically by stable identity. Optional ownership
        metadata is copied into the corresponding manifest record but is never
        interpreted here; reconstruction semantics remain owned by the
        explicit-ownership layer.

        :param records: Current immutable artifact-store snapshots.
        :param ownership_metadata: Keyword argument, optional, defaults to ``None``.
            Candidate-ID to JSON-safe explicit-reconstruction metadata.
        :param failure_diagnostics: Keyword argument, optional, defaults to ``()``.
            Current bounded failed-evaluation diagnostic records retained outside
            ``ArtifactStore``.
        :raises ArtifactProvenanceError: If records/metadata are malformed or the atomic
            manifest write fails.
        """
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ArtifactProvenanceError(
                "records must be a sequence of ArtifactRecord values"
            )
        normalized_records = tuple(records)
        if any(not isinstance(record, ArtifactRecord) for record in normalized_records):
            raise ArtifactProvenanceError(
                "records must contain only ArtifactRecord values"
            )
        candidate_ids = [record.candidate_id for record in normalized_records]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ArtifactProvenanceError(
                "manifest records contain duplicate candidate identities"
            )

        raw_ownership = {} if ownership_metadata is None else ownership_metadata
        if not isinstance(raw_ownership, Mapping):
            raise ArtifactProvenanceError(
                "ownership_metadata must be a mapping or None"
            )
        if any(not isinstance(candidate_id, str) for candidate_id in raw_ownership):
            raise ArtifactProvenanceError(
                "ownership_metadata keys must be candidate IDs"
            )
        unknown = sorted(set(raw_ownership).difference(candidate_ids))
        if unknown:
            raise ArtifactProvenanceError(
                "ownership_metadata contains unknown candidate identities: "
                + ", ".join(unknown)
            )
        ownership = {
            candidate_id: _json_copy(
                raw_ownership[candidate_id],
                context=f"ownership metadata for {candidate_id!r}",
            )
            for candidate_id in sorted(raw_ownership)
        }

        if not isinstance(failure_diagnostics, Sequence) or isinstance(
            failure_diagnostics, (str, bytes)
        ):
            raise ArtifactProvenanceError(
                "failure_diagnostics must be a sequence of mappings"
            )
        normalized_failures: list[dict[str, object]] = []
        for index, failure in enumerate(failure_diagnostics):
            if not isinstance(failure, Mapping):
                raise ArtifactProvenanceError(
                    "failure_diagnostics must contain only mappings"
                )
            normalized = _json_copy(failure, context=f"failure_diagnostics[{index}]")
            if not isinstance(normalized, dict):
                raise ArtifactProvenanceError(
                    "failure_diagnostics entries must normalize to JSON objects"
                )
            normalized_failures.append(normalized)
        normalized_failures.sort(
            key=lambda item: _canonical_json(
                item, context="failure diagnostic manifest record"
            )
        )

        manifest_records: list[dict[str, object]] = []
        for record in sorted(normalized_records, key=lambda item: item.candidate_id):
            candidate_state = record.candidate.to_state()
            manifest_records.append(
                {
                    "candidate_id": record.candidate_id,
                    "generation": record.candidate.generation,
                    "objective": record.candidate.objective,
                    "properties": candidate_state["properties"],
                    "lineage": candidate_state["lineage"],
                    "source_path": record.source_path,
                    "archive_path": record.archive_path,
                    "pins": [pin.value for pin in record.pins],
                    "retention_reasons": list(record.retention_reasons),
                    "status": record.status.value,
                    "ownership_metadata": ownership.get(record.candidate_id),
                }
            )
        manifest = {
            "version": _MANIFEST_VERSION,
            "calculation_context": self._calculation_context,
            "failure_diagnostics": normalized_failures,
            "records": manifest_records,
        }
        try:
            encoded = (
                json.dumps(
                    manifest,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactProvenanceError(
                "artifact manifest contains non-serializable state"
            ) from exc

        temporary = self._manifest_path.with_name(self._manifest_path.name + ".tmp")
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._manifest_path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ArtifactProvenanceError(
                f"could not write artifact manifest {self._manifest_path}"
            ) from exc
