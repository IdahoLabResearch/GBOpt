# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Provide shared lexical path normalization for artifact subsystem boundaries.

This private module owns only validation and lexical normalization of text filesystem
paths. Callers supply their layer-specific artifact exception type so cleanup and
provenance retain distinct public error vocabularies. Filesystem containment, deletion,
and provenance persistence remain outside this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from GBOpt.artifacts.types import ArtifactError


def _normalize_path(
    value: object,
    *,
    name: str,
    error_type: type[ArtifactError],
) -> Path:
    """Normalize one non-empty text filesystem path without resolving symlinks.

    :param value: Path-like value to normalize.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :param error_type: Keyword argument, required. Layer-specific artifact exception
        type raised for invalid path values.
    :return: Absolute lexical path.
    :raises ArtifactError: If ``value`` is not a non-empty text path-like value.
    """
    if not isinstance(value, (str, os.PathLike)):
        raise error_type(f"{name} must be a non-empty path-like value")
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise error_type(f"{name} must be a non-empty path-like value") from exc
    if isinstance(raw, bytes):
        raise error_type(f"{name} must use a text filesystem path")
    if not raw.strip():
        raise error_type(f"{name} must be a non-empty path-like value")
    return Path(os.path.abspath(raw))
