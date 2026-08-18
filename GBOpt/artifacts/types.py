# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define immutable artifact-retention domain values and validation boundaries.

This module consumes candidate identity, normalized retention properties, and validated
relaxed physical arrays. It returns detached value objects used by retention rules and
the runtime artifact store. Optimizer selection, filesystem cleanup, and external
structure serialization do not belong here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

import numpy as np


class ArtifactError(Exception):
    """Base class for artifact-subsystem errors."""


class ArtifactValueError(ArtifactError, ValueError):
    """Raised when artifact-domain state is malformed."""


RetentionValue: TypeAlias = (
    int | float | str | bool | tuple["RetentionValue", ...]
)

BUILTIN_PROPERTY_NAMES = frozenset(
    {
        "objective",
        "atom_count",
        "composition",
        "cell_volume",
        "candidate_id",
        "generation",
    }
)

__all__ = [
    "ArtifactError",
    "ArtifactPin",
    "ArtifactRecord",
    "ArtifactStatus",
    "ArtifactValueError",
    "BUILTIN_PROPERTY_NAMES",
    "CandidatePropertyContext",
    "RetentionCandidate",
    "RetentionValue",
]


def _require_nonempty_string(value: object, name: str) -> str:
    """Normalize a non-empty string field.

    :param value: Candidate string value.
    :param name: Field name for diagnostics.
    :return: Validated string.
    :raises ArtifactValueError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValueError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    """Normalize a non-Boolean non-negative integer.

    :param value: Candidate integer value.
    :param name: Field name for diagnostics.
    :return: Python integer.
    :raises ArtifactValueError: If the value is not a non-negative integer.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ArtifactValueError(f"{name} must be a non-Boolean integer")
    normalized = int(value)
    if normalized < 0:
        raise ArtifactValueError(f"{name} must be non-negative")
    return normalized


def _require_finite_real(value: object, name: str) -> float:
    """Normalize a finite non-Boolean real scalar.

    :param value: Candidate real value.
    :param name: Field name for diagnostics.
    :return: Python float.
    :raises ArtifactValueError: If the value is not a finite real scalar.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ArtifactValueError(f"{name} must be a non-Boolean real scalar")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(normalized):
        raise ArtifactValueError(f"{name} must be finite")
    return normalized


def normalize_retention_value(value: object, *, name: str = "retention value") -> RetentionValue:
    """Normalize one supported retention-property value.

    Supported values are immutable scalar values and recursively nested tuples. NumPy
    scalar values are normalized to their Python equivalents. Mutable containers and
    arrays are deliberately rejected.

    :param value: Value supplied by built-in or user property acquisition.
    :param name: Keyword argument, optional, defaults to ``"retention value"``.
        Diagnostic field name.
    :return: JSON-friendly immutable retention value.
    :raises ArtifactValueError: If the value is unsupported or a numeric value is not
        finite.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (Integral, np.integer)):
        return int(value)
    if isinstance(value, (Real, np.floating)):
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ArtifactValueError(f"{name} must be finite")
        return normalized
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return tuple(
            normalize_retention_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ArtifactValueError(
        f"{name} has unsupported type {type(value).__name__}; expected int, float, "
        "str, bool, or tuple"
    )


def retention_value_to_state(value: RetentionValue) -> object:
    """Convert one normalized retention value to plain JSON-safe state.

    Tuples are encoded as lists. Restore paths use :func:`retention_value_from_state`
    so provider-facing normalization can continue to reject mutable lists.

    :param value: Normalized retention value.
    :return: JSON-safe scalar or list state.
    """
    if isinstance(value, tuple):
        return [retention_value_to_state(item) for item in value]
    return value


def retention_value_from_state(value: object, *, name: str = "retention value") -> RetentionValue:
    """Restore one retention value from JSON-decoded state.

    :param value: JSON-decoded scalar or list.
    :param name: Keyword argument, optional, defaults to ``"retention value"``.
        Diagnostic field name.
    :return: Normalized immutable retention value.
    :raises ArtifactValueError: If the serialized value is malformed.
    """
    if isinstance(value, list):
        return tuple(
            retention_value_from_state(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    return normalize_retention_value(value, name=name)


def normalize_property_mapping(
    properties: Mapping[object, object],
    *,
    reject_reserved: bool = False,
) -> Mapping[str, RetentionValue]:
    """Validate and freeze a property mapping.

    :param properties: Property-name to value mapping.
    :param reject_reserved: Keyword argument, optional, defaults to ``False``. Reject
        built-in names when validating user-provider output.
    :return: Read-only mapping with normalized values and lexical key order.
    :raises ArtifactValueError: If the mapping, a key, or a value is invalid.
    """
    if not isinstance(properties, Mapping):
        raise ArtifactValueError("properties must be a mapping")
    keys = tuple(properties)
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ArtifactValueError("property names must be non-empty strings")
    normalized: dict[str, RetentionValue] = {}
    for key in sorted(keys):
        if reject_reserved and key in BUILTIN_PROPERTY_NAMES:
            raise ArtifactValueError(
                f"property provider may not overwrite reserved property {key!r}"
            )
        normalized[key] = normalize_retention_value(
            properties[key], name=f"property {key!r}"
        )
    return MappingProxyType(normalized)


def _readonly_array(value: object, *, name: str) -> np.ndarray:
    """Return an owned read-only NumPy copy without coercing its dtype.

    :param value: Array-like input.
    :param name: Keyword argument, required. Diagnostic field name.
    :return: Read-only NumPy array.
    :raises ArtifactValueError: If conversion fails.
    """
    try:
        result = np.array(value, copy=True)
    except (TypeError, ValueError) as exc:
        raise ArtifactValueError(f"{name} must be array-like") from exc
    result.setflags(write=False)
    return result


def _readonly_finite_real_array(value: object, *, name: str) -> np.ndarray:
    """Return an owned read-only float array from real numeric input.

    String, object, complex, and Boolean arrays are rejected rather than silently
    coerced because physical coordinates and box bounds are public domain inputs.

    :param value: Numeric array-like input.
    :param name: Keyword argument, required. Diagnostic field name.
    :return: Read-only floating-point array containing only finite values.
    :raises ArtifactValueError: If the input is not a finite real numeric array.
    """
    raw = _readonly_array(value, name=name)
    if raw.dtype.kind not in ("i", "u", "f"):
        raise ArtifactValueError(f"{name} must contain real numeric values")
    try:
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactValueError(f"{name} must contain finite real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ArtifactValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _normalize_grain_labels(value: object, *, expected_count: int) -> np.ndarray:
    """Return strict read-only left/right grain labels aligned with atom rows.

    :param value: Candidate grain-label array.
    :param expected_count: Keyword argument, required. Required label count.
    :return: Read-only ``int8`` array containing only labels 0 and 1.
    :raises ArtifactValueError: If shape, dtype, or label values are invalid.
    """
    raw = _readonly_array(value, name="grain_labels")
    if raw.shape != (expected_count,):
        raise ArtifactValueError(
            "grain_labels must be one-dimensional and aligned with atoms"
        )
    if raw.dtype.kind not in ("i", "u"):
        raise ArtifactValueError("grain_labels must use an integer dtype")
    if not np.all(np.isin(raw, (0, 1))):
        raise ArtifactValueError("grain_labels must contain only left/right labels 0 and 1")
    result = np.array(raw, dtype=np.int8, copy=True)
    result.setflags(write=False)
    return result


def _validate_atom_rows(value: object) -> np.ndarray:
    """Return detached structured atom rows with validated names and coordinates.

    :param value: Relaxed structured atom array.
    :return: Read-only structured array with the caller's field layout preserved.
    :raises ArtifactValueError: If required fields, coordinate values, or names are invalid.
    """
    atoms = _readonly_array(value, name="atoms")
    if atoms.ndim != 1 or atoms.dtype.names is None:
        raise ArtifactValueError("atoms must be a one-dimensional structured array")
    required_fields = ("name", "x", "y", "z")
    missing = [field for field in required_fields if field not in atoms.dtype.names]
    if missing:
        raise ArtifactValueError(
            "atoms must contain structured fields: " + ", ".join(required_fields)
        )
    names = atoms["name"]
    if names.dtype.kind not in ("U", "O"):
        raise ArtifactValueError("atoms 'name' field must contain strings")
    if any(not isinstance(name, (str, np.str_)) or not str(name).strip() for name in names):
        raise ArtifactValueError("atoms 'name' values must be non-empty strings")
    for coordinate in ("x", "y", "z"):
        values = atoms[coordinate]
        if values.dtype.kind not in ("i", "u", "f"):
            raise ArtifactValueError(f"atoms {coordinate!r} field must be real numeric")
        if not np.all(np.isfinite(values)):
            raise ArtifactValueError(f"atoms {coordinate!r} coordinates must be finite")
    return atoms


@dataclass(frozen=True, slots=True)
class RetentionCandidate:
    """Narrow immutable candidate view consumed by retention rules.

    :param candidate_id: Stable logical candidate identity, independent of artifact
        paths.
    :param generation: Non-negative optimizer generation index.
    :param objective: Finite optimizer objective value.
    :param properties: Immutable property namespace available to retention rules.
    :param lineage: Stable logical parent identities.
    :raises ArtifactValueError: If identity, generation, objective, properties, or
        lineage violate the candidate contract.
    """

    candidate_id: str
    generation: int
    objective: float
    properties: Mapping[str, RetentionValue] = field(default_factory=dict)
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate scalar state and freeze the property namespace.

        :raises ArtifactValueError: If identity, generation, objective, properties, or
            lineage violate the retention-candidate contract.
        """
        candidate_id = _require_nonempty_string(self.candidate_id, "candidate_id")
        generation = _require_nonnegative_int(self.generation, "generation")
        objective = _require_finite_real(self.objective, "objective")
        properties = dict(normalize_property_mapping(self.properties))

        builtins: dict[str, RetentionValue] = {
            "candidate_id": candidate_id,
            "generation": generation,
            "objective": objective,
        }
        for key, expected in builtins.items():
            if key in properties and properties[key] != expected:
                raise ArtifactValueError(
                    f"property {key!r} conflicts with the candidate field value"
                )
            properties[key] = expected

        if not isinstance(self.lineage, tuple):
            raise ArtifactValueError("lineage must be a tuple of candidate IDs")
        lineage = tuple(
            _require_nonempty_string(parent_id, f"lineage[{index}]")
            for index, parent_id in enumerate(self.lineage)
        )

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "properties",
            MappingProxyType(dict(sorted(properties.items()))),
        )
        object.__setattr__(self, "lineage", lineage)

    def property_value(self, name: str) -> RetentionValue:
        """Return one named property.

        :param name: Property name.
        :return: Normalized property value.
        :raises KeyError: If the property is absent.
        """
        if name not in self.properties:
            raise KeyError(name)
        return self.properties[name]

    def to_state(self) -> dict[str, object]:
        """Return deterministic JSON-safe candidate state.

        :return: Callback-free candidate state suitable for JSON serialization.
        """
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "objective": self.objective,
            "properties": {
                key: retention_value_to_state(value)
                for key, value in self.properties.items()
                if key not in {"candidate_id", "generation", "objective"}
            },
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_state(cls, state: object) -> "RetentionCandidate":
        """Restore a candidate from JSON-decoded state.

        :param state: Serialized candidate dictionary.
        :return: Validated candidate value object.
        :raises ArtifactValueError: If state is malformed.
        """
        if not isinstance(state, dict):
            raise ArtifactValueError("retention candidate state must be a dictionary")
        try:
            raw_properties = state["properties"]
            raw_lineage = state["lineage"]
            candidate_id = state["candidate_id"]
            generation = state["generation"]
            objective = state["objective"]
        except KeyError as exc:
            raise ArtifactValueError("retention candidate state is incomplete") from exc
        if not isinstance(raw_properties, dict):
            raise ArtifactValueError("retention candidate properties state is invalid")
        if not isinstance(raw_lineage, list):
            raise ArtifactValueError("retention candidate lineage state is invalid")
        properties = {
            key: retention_value_from_state(value, name=f"property {key!r}")
            for key, value in raw_properties.items()
        }
        return cls(
            candidate_id=candidate_id,
            generation=generation,
            objective=objective,
            properties=properties,
            lineage=tuple(raw_lineage),
        )


@dataclass(frozen=True, slots=True)
class CandidatePropertyContext:
    """Validated relaxed physical state supplied to a property provider.

    :param candidate_id: Stable logical candidate identity.
    :param generation: Non-negative optimizer generation index.
    :param objective: Finite optimizer objective value.
    :param atoms: Relaxed structured atom rows.
    :param box_dims: Finite 3 by 2 orthogonal box bounds.
    :param gb_plane_x: Finite grain-boundary plane coordinate.
    :param grain_labels: Optional explicit grain labels aligned with atom rows.
    :raises ArtifactValueError: If scalar or physical candidate state is malformed.
    """

    candidate_id: str
    generation: int
    objective: float
    atoms: np.ndarray = field(repr=False)
    box_dims: np.ndarray = field(repr=False)
    gb_plane_x: float
    grain_labels: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate and detach physical candidate arrays from live optimizer state.

        :raises ArtifactValueError: If identity, scalar state, atom rows, box bounds,
            grain labels, or the grain-boundary plane are malformed.
        """
        candidate_id = _require_nonempty_string(self.candidate_id, "candidate_id")
        generation = _require_nonnegative_int(self.generation, "generation")
        objective = _require_finite_real(self.objective, "objective")
        atoms = _validate_atom_rows(self.atoms)

        box_dims = _readonly_finite_real_array(self.box_dims, name="box_dims")
        if box_dims.shape != (3, 2):
            raise ArtifactValueError("box_dims must be a finite 3 by 2 array")
        if np.any(box_dims[:, 1] <= box_dims[:, 0]):
            raise ArtifactValueError("box_dims upper bounds must exceed lower bounds")

        grain_labels = self.grain_labels
        if grain_labels is not None:
            grain_labels = _normalize_grain_labels(
                grain_labels, expected_count=len(atoms)
            )

        gb_plane_x = _require_finite_real(self.gb_plane_x, "gb_plane_x")
        if not box_dims[0, 0] < gb_plane_x < box_dims[0, 1]:
            raise ArtifactValueError(
                "gb_plane_x must lie strictly inside the x box bounds"
            )

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "box_dims", box_dims)
        object.__setattr__(self, "grain_labels", grain_labels)
        object.__setattr__(self, "gb_plane_x", gb_plane_x)


class ArtifactPin(str, Enum):
    """Operational reasons that make a candidate artifact restart-critical."""

    ACTIVE_POPULATION = "active_population"
    CANDIDATE_CHECKPOINT = "candidate_checkpoint"
    RUN_CHECKPOINT = "run_checkpoint"
    BEST_RESULT = "best_result"
    CARRYOVER_CACHE = "carryover_cache"


class ArtifactStatus(str, Enum):
    """Reference-derived status of a registered candidate artifact."""

    UNREFERENCED = "unreferenced"
    PINNED = "pinned"
    RETAINED = "retained"
    PINNED_AND_RETAINED = "pinned_and_retained"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Immutable snapshot of one candidate's runtime artifact references.

    :param candidate: Scientific candidate metadata.
    :param source_path: Optional evaluator artifact path. The candidate identity remains
        independent of this path.
    :param pins: Active operational pins.
    :param retention_reasons: Active scientific retention reasons.
    :raises ArtifactValueError: If candidate state, source path, pins, or reasons are
        malformed.
    """

    candidate: RetentionCandidate
    source_path: str | None = None
    pins: tuple[ArtifactPin, ...] = ()
    retention_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize path and reference collections deterministically.

        :raises ArtifactValueError: If candidate state, source path, pins, or retention
            reasons are malformed.
        """
        if not isinstance(self.candidate, RetentionCandidate):
            raise ArtifactValueError("candidate must be a RetentionCandidate")
        source_path = self.source_path
        if source_path is not None:
            if (
                not isinstance(source_path, (str, Path))
                or not str(source_path).strip()
            ):
                raise ArtifactValueError("source_path must be a non-empty path or None")
            source_path = str(source_path)

        pins: set[ArtifactPin] = set()
        for pin in self.pins:
            if not isinstance(pin, ArtifactPin):
                raise ArtifactValueError("pins must contain ArtifactPin values")
            pins.add(pin)

        reasons = {
            _require_nonempty_string(reason, "retention reason")
            for reason in self.retention_reasons
        }

        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "pins", tuple(sorted(pins, key=lambda pin: pin.value)))
        object.__setattr__(self, "retention_reasons", tuple(sorted(reasons)))

    @property
    def candidate_id(self) -> str:
        """Return the stable logical candidate identity.

        :return: Candidate identity independent of artifact paths.
        """
        return self.candidate.candidate_id

    @property
    def status(self) -> ArtifactStatus:
        """Return status derived from independent operational and scientific refs.

        :return: Reference-derived artifact status.
        """
        if self.pins and self.retention_reasons:
            return ArtifactStatus.PINNED_AND_RETAINED
        if self.pins:
            return ArtifactStatus.PINNED
        if self.retention_reasons:
            return ArtifactStatus.RETAINED
        return ArtifactStatus.UNREFERENCED
