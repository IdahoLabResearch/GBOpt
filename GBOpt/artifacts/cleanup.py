# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Provide constrained, backend-aware artifact cleanup primitives.

This module owns validation of filesystem paths that GBOpt is explicitly allowed to
remove and the runtime dispatch boundary for evaluator-owned cleanup callbacks. It
consumes logical candidate/source metadata after the optimizer has declared an artifact
transient. Retention decisions, checkpoint commits, and backend-specific work-directory
layout remain outside this module.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from GBOpt.artifacts.types import ArtifactError

__all__ = [
    "ArtifactCleanupError",
    "ArtifactCleanupRequest",
    "remove_managed_path",
]


class ArtifactCleanupError(ArtifactError):
    """Raised when artifact cleanup configuration or execution is unsafe or invalid."""


def _normalize_path(value: object, *, name: str) -> Path:
    """Normalize one non-empty filesystem path without resolving symlinks.

    :param value: Path-like value to normalize.
    :param name: Keyword argument, required. Argument name used in diagnostics.
    :return: Absolute lexical path.
    :raises ArtifactCleanupError: If ``value`` is not a non-empty path-like value.
    """
    if not isinstance(value, (str, os.PathLike)):
        raise ArtifactCleanupError(f"{name} must be a non-empty path-like value")
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ArtifactCleanupError(
            f"{name} must be a non-empty path-like value"
        ) from exc
    if isinstance(raw, bytes):
        raise ArtifactCleanupError(f"{name} must use a text filesystem path")
    if not raw.strip():
        raise ArtifactCleanupError(f"{name} must be a non-empty path-like value")
    return Path(os.path.abspath(raw))


def _canonical_path(path: Path, *, name: str) -> Path:
    """Resolve a path for containment validation without requiring it to exist.

    :param path: Absolute lexical path to resolve.
    :param name: Keyword argument, required. Argument name used in diagnostics.
    :return: Canonical path with existing symlinks resolved.
    :raises ArtifactCleanupError: If canonicalization fails, including symlink loops.
    """
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ArtifactCleanupError(f"could not canonicalize {name} {path}") from exc


def _validated_managed_target(
    path: str | os.PathLike[str],
    *,
    managed_root: str | os.PathLike[str],
) -> Path:
    """Return a deletion target proven to remain beneath a managed root.

    The target itself may not be the managed root. Canonical containment rejects both
    lexical ``..`` escapes and symlink-based escapes. The returned path preserves the
    caller's lexical target so deleting an in-root symlink removes the symlink rather
    than the file to which it points.

    :param path: Candidate file or directory to remove.
    :param managed_root: Keyword argument, required. Root whose descendants GBOpt is
        explicitly allowed to remove.
    :return: Absolute lexical target after canonical containment validation.
    :raises ArtifactCleanupError: If either path is malformed, the target resolves
        outside ``managed_root``, or the target resolves to the managed root itself.
    """
    root = _normalize_path(managed_root, name="managed_root")
    target = _normalize_path(path, name="path")
    canonical_root = _canonical_path(root, name="managed_root")
    canonical_target = _canonical_path(target, name="path")
    if canonical_target == canonical_root:
        raise ArtifactCleanupError("refusing to remove the managed artifact root itself")
    try:
        canonical_target.relative_to(canonical_root)
    except ValueError as exc:
        raise ArtifactCleanupError(
            f"refusing to remove path outside managed artifact root: {target}"
        ) from exc
    return target


def remove_managed_path(
    path: str | os.PathLike[str],
    *,
    managed_root: str | os.PathLike[str],
) -> None:
    """Remove one file, symlink, or directory beneath an explicitly managed root.

    Missing targets are treated as already cleaned. Directory deletion is recursive, but
    the managed root itself and canonical paths outside it are never removed.

    :param path: File, symlink, or directory that has become transient.
    :param managed_root: Keyword argument, required. Root whose descendants GBOpt is
        explicitly allowed to remove.
    :raises ArtifactCleanupError: If containment validation fails or filesystem removal
        fails.
    """
    target = _validated_managed_target(path, managed_root=managed_root)
    try:
        if target.is_symlink():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    except OSError as exc:
        raise ArtifactCleanupError(f"failed to remove managed artifact {target}") from exc


@dataclass(frozen=True, slots=True)
class ArtifactCleanupRequest:
    """Describe one evaluator source artifact that is safe to clean after commit.

    A backend callback may use ``candidate_id`` and ``source_path`` to identify a richer
    work directory and may use ``archive_path`` to confirm where a retained canonical
    structure was materialized before removing evaluator-owned files.

    :param candidate_id: Stable logical candidate identity.
    :param source_path: Evaluator-returned source structure path.
    :param archive_path: Canonical retained structure path, when one exists.
    :raises ArtifactCleanupError: If identity or path fields are malformed.
    """

    candidate_id: str
    source_path: Path
    archive_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate and normalize callback-facing cleanup metadata.

        :raises ArtifactCleanupError: If identity or path fields are malformed.
        """
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ArtifactCleanupError("candidate_id must be a non-empty string")
        source_path = _normalize_path(self.source_path, name="source_path")
        archive_path = None
        if self.archive_path is not None:
            archive_path = _normalize_path(self.archive_path, name="archive_path")
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "archive_path", archive_path)


class _ArtifactCleaner:
    """Dispatch transient evaluator-source cleanup through one explicit ownership mode.

    ``managed_artifact_root`` lets GBOpt remove the exact evaluator-returned path only
    after containment validation. ``cleanup_candidate`` delegates richer backend layout
    semantics to the evaluator/application. The modes are mutually exclusive so there is
    one unambiguous owner for evaluator-source deletion.

    :param managed_artifact_root: Optional root containing GBOpt-managed evaluator paths.
    :param cleanup_candidate: Optional evaluator-owned cleanup callback.
    :raises ArtifactCleanupError: If configuration is malformed or ambiguous.
    """

    def __init__(
        self,
        *,
        managed_artifact_root: str | os.PathLike[str] | None = None,
        cleanup_candidate: Callable[[ArtifactCleanupRequest], None] | None = None,
    ) -> None:
        """Validate runtime cleanup configuration.

        :param managed_artifact_root: Keyword argument, optional, defaults to ``None``.
            Root containing evaluator paths GBOpt may remove directly.
        :param cleanup_candidate: Keyword argument, optional, defaults to ``None``.
            Backend callback invoked for each transient evaluator source.
        :raises ArtifactCleanupError: If both modes are configured or either supplied
            value has an invalid type/value.
        """
        if managed_artifact_root is not None and cleanup_candidate is not None:
            raise ArtifactCleanupError(
                "configure either managed_artifact_root or cleanup_candidate, not both"
            )
        if cleanup_candidate is not None and not callable(cleanup_candidate):
            raise ArtifactCleanupError("cleanup_candidate must be callable or None")
        self._managed_root = (
            None
            if managed_artifact_root is None
            else _normalize_path(managed_artifact_root, name="managed_artifact_root")
        )
        self._cleanup_candidate = cleanup_candidate

    @property
    def enabled(self) -> bool:
        """Return whether evaluator-source cleanup has an explicit owner.

        :return: Whether a managed root or cleanup callback is configured.
        """
        return self._managed_root is not None or self._cleanup_candidate is not None

    def cleanup_source(self, request: ArtifactCleanupRequest) -> None:
        """Clean one transient evaluator source through the configured ownership mode.

        :param request: Committed transient evaluator-source metadata.
        :raises ArtifactCleanupError: If ``request`` is invalid, no cleanup owner is
            configured, containment validation fails, or the backend callback/removal
            fails.
        """
        if not isinstance(request, ArtifactCleanupRequest):
            raise ArtifactCleanupError("request must be an ArtifactCleanupRequest")
        if self._cleanup_candidate is not None:
            try:
                self._cleanup_candidate(request)
            except Exception as exc:
                # External cleanup callbacks are a deliberate fault-containment boundary:
                # any backend failure must become a recoverable storage leak upstream.
                raise ArtifactCleanupError(
                    f"cleanup callback failed for candidate {request.candidate_id!r}"
                ) from exc
            return
        if self._managed_root is not None:
            remove_managed_path(request.source_path, managed_root=self._managed_root)
            return
        raise ArtifactCleanupError("no evaluator artifact cleanup owner is configured")
