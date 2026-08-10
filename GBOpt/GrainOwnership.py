"""Persistent left/right grain ownership for interface candidates.

This module owns immutable optimizer-side grain identity and the interface geometry
required to interpret it. External file syntax, calculator execution, and optimizer
selection policy do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    normalize_boundary_normal_topology,
)

LEFT_GRAIN_LABEL = 0
RIGHT_GRAIN_LABEL = 1
_SUPPORTED_LABELS = frozenset((LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL))
_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


class GrainOwnershipError(ValueError):
    """Raised when explicit persistent grain ownership metadata is malformed."""


def _readonly_copy(
    values: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Return an independent read-only NumPy array.

    :param values: Source array whose values will be copied.
    :param dtype: Keyword argument, optional, defaults to ``None``. Requested result
        dtype.
    :return: A copied array with mutation disabled.
    """
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _strict_int(name: str, value: object) -> int:
    """Validate one integer representable by the ownership storage dtype.

    :param name: Field name used in validation errors.
    :param value: Value to validate.
    :return: A normalized Python integer.
    :raises GrainOwnershipError: If the value is Boolean, nonintegral, or outside the
        supported signed ``int64`` range.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GrainOwnershipError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < _INT64_MIN or normalized > _INT64_MAX:
        raise GrainOwnershipError(
            f"{name} must be representable as a signed 64-bit integer"
        )
    return normalized


def _strict_finite_real(name: str, value: object) -> float:
    """Validate one finite real scalar without accepting Boolean coercion.

    :param name: Field name used in validation errors.
    :param value: Value to validate.
    :return: A normalized Python float.
    :raises GrainOwnershipError: If the value is Boolean, non-real, or non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GrainOwnershipError(f"{name} must be a real scalar")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise GrainOwnershipError(f"{name} must be finite")
    return normalized


def _strict_boolean_pair(name: str, value: object) -> tuple[bool, bool]:
    """Validate an explicit pair of Boolean flags.

    :param name: Field name used in validation errors.
    :param value: Pair to validate.
    :return: Two normalized Python booleans.
    :raises GrainOwnershipError: If the value is not exactly two Boolean flags.
    """
    try:
        values = tuple(value)  # type: ignore[ty:invalid-argument-type]
    except TypeError as exc:
        raise GrainOwnershipError(
            f"{name} must contain exactly two Boolean flags"
        ) from exc
    if len(values) != 2 or not all(
        isinstance(item, (bool, np.bool_)) for item in values
    ):
        raise GrainOwnershipError(
            f"{name} must contain exactly two Boolean flags"
        )
    return bool(values[0]), bool(values[1])


def _strict_optional_boolean(name: str, value: object) -> bool | None:
    """Validate an optional Boolean compatibility flag.

    :param name: Field name used in validation errors.
    :param value: Value to validate.
    :return: ``None`` or a normalized Python boolean.
    :raises GrainOwnershipError: If a non-Boolean value is supplied.
    """
    if value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise GrainOwnershipError(f"{name} must be a Boolean or None")
    return bool(value)


def _strict_x_bounds(name: str, value: object) -> np.ndarray:
    """Validate a strictly ordered pair of finite x coordinates.

    :param name: Field name used in validation errors.
    :param value: Bounds to validate.
    :return: A two-element floating-point array.
    :raises GrainOwnershipError: If the bounds are malformed or unordered.
    """
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2,):
        raise GrainOwnershipError(f"{name} must contain two finite values")
    bounds = np.asarray(
        [
            _strict_finite_real(f"{name}[0]", raw[0]),
            _strict_finite_real(f"{name}[1]", raw[1]),
        ],
        dtype=float,
    )
    if bounds[0] >= bounds[1]:
        raise GrainOwnershipError(f"{name} must be strictly ordered")
    return bounds


@dataclass(frozen=True, slots=True, init=False)
class GrainOwnership:
    """Immutable left/right labels and interface geometry for one candidate.

    ``atom_ids`` are serialization-local alignment keys. Persistent grain identity is
    represented only by ``labels``. The physical grain bounds may be separated by an
    empty interval containing ``gb_plane_x``.
    """

    _atom_ids: np.ndarray
    _labels: np.ndarray
    gb_plane_x: float
    inplane_periodic: tuple[bool, bool]
    _left_grain_x_bounds: np.ndarray | None
    _right_grain_x_bounds: np.ndarray
    coordinate_tolerance: float
    _normal_topology: BoundaryNormalTopology

    def __init__(
        self,
        *,
        atom_ids: np.ndarray,
        labels: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        periodic_outer_x_interface: bool | None = None,
        left_grain_x_bounds: np.ndarray | tuple[float, float] | None = None,
        normal_topology: BoundaryNormalTopology | str | None = None,
    ) -> None:
        """Construct immutable explicit grain ownership metadata.

        :param atom_ids: Keyword argument, required. Positive, unique
            serialization-local atom IDs.
        :param labels: Keyword argument, required. Left/right grain labels aligned with
            ``atom_ids``.
        :param gb_plane_x: Keyword argument, required. Nominal central interface plane
            in angstroms.
        :param inplane_periodic: Keyword argument, required. Explicit y/z periodicity
            flags.
        :param right_grain_x_bounds: Keyword argument, required. Physical right-grain x
            bounds in angstroms.
        :param coordinate_tolerance: Keyword argument, required. Positive coordinate
            tolerance in angstroms.
        :param periodic_outer_x_interface: Keyword argument, optional, defaults to
            ``None``. Legacy topology compatibility flag.
        :param left_grain_x_bounds: Keyword argument, optional, defaults to ``None``.
            Physical left-grain x bounds in angstroms.
        :param normal_topology: Keyword argument, optional, defaults to ``None``.
            Explicit boundary-normal topology.
        :raises GrainOwnershipError: If any ownership or geometry invariant is invalid.
        """
        raw_ids = np.asarray(atom_ids, dtype=object)
        raw_labels = np.asarray(labels, dtype=object)
        if raw_ids.ndim != 1 or raw_labels.ndim != 1:
            raise GrainOwnershipError("atom_ids and labels must be one-dimensional")
        if raw_ids.size != raw_labels.size:
            raise GrainOwnershipError(
                "ownership-array length must equal atom ID count"
            )

        normalized_ids = np.empty(raw_ids.size, dtype=np.int64)
        for index, value in enumerate(raw_ids.tolist()):
            normalized_ids[index] = _strict_int("atom ID", value)
        if np.any(normalized_ids <= 0):
            raise GrainOwnershipError("atom IDs must be positive")
        if np.unique(normalized_ids).size != normalized_ids.size:
            raise GrainOwnershipError("atom IDs must be unique")

        normalized_labels = np.empty(raw_labels.size, dtype=np.int8)
        for index, value in enumerate(raw_labels.tolist()):
            parsed = _strict_int("grain label", value)
            if parsed not in _SUPPORTED_LABELS:
                raise GrainOwnershipError(
                    "grain labels must be exactly 0 (left) or 1 (right)"
                )
            normalized_labels[index] = parsed

        plane = _strict_finite_real("gb_plane_x", gb_plane_x)
        tolerance = _strict_finite_real("coordinate_tolerance", coordinate_tolerance)
        if tolerance <= 0.0:
            raise GrainOwnershipError("coordinate_tolerance must be positive")
        periodic = _strict_boolean_pair("inplane_periodic", inplane_periodic)
        legacy_outer_interface = _strict_optional_boolean(
            "periodic_outer_x_interface", periodic_outer_x_interface
        )

        right_bounds = _strict_x_bounds(
            "right_grain_x_bounds", right_grain_x_bounds
        )
        if right_bounds[0] < plane - tolerance:
            raise GrainOwnershipError(
                "right-grain lower bound must be on or to the right of gb_plane_x"
            )

        normalized_left: np.ndarray | None
        if left_grain_x_bounds is None:
            normalized_left = None
        else:
            normalized_left = _strict_x_bounds(
                "left_grain_x_bounds", left_grain_x_bounds
            )
            if normalized_left[1] > plane + tolerance:
                raise GrainOwnershipError(
                    "left-grain upper bound must be on or to the left of gb_plane_x"
                )

        try:
            topology = normalize_boundary_normal_topology(normal_topology)
        except ValueError as exc:
            raise GrainOwnershipError(str(exc)) from exc
        if legacy_outer_interface is True:
            if topology is BoundaryNormalTopology.UNKNOWN:
                topology = BoundaryNormalTopology.PERIODIC_BICRYSTAL
            elif topology is not BoundaryNormalTopology.PERIODIC_BICRYSTAL:
                raise GrainOwnershipError(
                    "periodic_outer_x_interface=True conflicts with normal_topology"
                )
        elif (
            legacy_outer_interface is False
            and topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
        ):
            raise GrainOwnershipError(
                "periodic_outer_x_interface=False conflicts with periodic topology"
            )

        object.__setattr__(self, "_atom_ids", _readonly_copy(normalized_ids))
        object.__setattr__(self, "_labels", _readonly_copy(normalized_labels))
        object.__setattr__(self, "gb_plane_x", plane)
        object.__setattr__(self, "inplane_periodic", periodic)
        object.__setattr__(
            self,
            "_left_grain_x_bounds",
            None if normalized_left is None else _readonly_copy(normalized_left),
        )
        object.__setattr__(self, "_right_grain_x_bounds", _readonly_copy(right_bounds))
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(self, "_normal_topology", topology)

    @classmethod
    def from_interface_candidate(
        cls,
        candidate: Any,
        *,
        atom_ids: np.ndarray | None = None,
    ) -> GrainOwnership:
        """Build explicit ownership from an interface-candidate value object.

        :param candidate: Interface-candidate-like object providing atoms, labels,
            geometry, periodicity, and topology.
        :param atom_ids: Keyword argument, optional, defaults to ``None``. When omitted,
            consecutive IDs are assigned in candidate row order.
        :return: Immutable ownership metadata aligned with the supplied or generated
            atom IDs.
        :raises GrainOwnershipError: If candidate metadata or atom IDs are invalid.
        """
        atoms = np.asarray(candidate.atoms)
        if atom_ids is None:
            atom_ids = np.arange(1, atoms.size + 1, dtype=np.int64)
        return cls(
            atom_ids=atom_ids,
            labels=candidate.grain_labels,
            gb_plane_x=candidate.gb_plane_x,
            inplane_periodic=candidate.inplane_periodic,
            left_grain_x_bounds=candidate.left_grain_x_bounds,
            right_grain_x_bounds=candidate.right_grain_x_bounds,
            coordinate_tolerance=candidate.coordinate_tolerance,
            normal_topology=candidate.normal_topology,
        )

    @property
    def atom_ids(self) -> np.ndarray:
        """Defensive copy of serialization-local atom IDs."""
        return _readonly_copy(self._atom_ids)

    @property
    def labels(self) -> np.ndarray:
        """Defensive copy of persistent grain labels."""
        return _readonly_copy(self._labels)

    @property
    def left_grain_x_bounds(self) -> np.ndarray | None:
        """Return optional physical left-grain x bounds.

        :return: A read-only copy of the bounds, or ``None`` when unavailable.
        """
        if self._left_grain_x_bounds is None:
            return None
        return _readonly_copy(self._left_grain_x_bounds)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        """Physical right-grain x bounds."""
        return _readonly_copy(self._right_grain_x_bounds)

    @property
    def normal_topology(self) -> BoundaryNormalTopology:
        """Explicit boundary-normal topology."""
        return self._normal_topology

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Whether the outer x boundary represents a periodic interface."""
        return self._normal_topology.periodic_outer_x_interface

    def aligned_to(self, atom_ids: np.ndarray) -> GrainOwnership:
        """Return ownership reordered to supplied file-row atom IDs.

        :param atom_ids: Atom IDs in the row order of a loaded structure file.
        :return: New immutable ownership metadata aligned to the supplied row order.
        :raises GrainOwnershipError: If IDs are malformed or do not exactly match the
            ownership ID set.
        """
        requested = np.asarray(atom_ids, dtype=object)
        if requested.ndim != 1:
            raise GrainOwnershipError("loaded atom IDs must be one-dimensional")
        normalized = np.empty(requested.size, dtype=np.int64)
        for index, value in enumerate(requested.tolist()):
            normalized[index] = _strict_int("loaded atom ID", value)
        if np.any(normalized <= 0) or np.unique(normalized).size != normalized.size:
            raise GrainOwnershipError("loaded atom IDs must be positive and unique")
        if normalized.size != self._atom_ids.size:
            raise GrainOwnershipError("loaded atom ID count does not match ownership")

        expected_order = np.argsort(self._atom_ids, kind="stable")
        expected_ids = self._atom_ids[expected_order]
        loaded_order = np.argsort(normalized, kind="stable")
        if not np.array_equal(normalized[loaded_order], expected_ids):
            raise GrainOwnershipError("loaded atom IDs do not match ownership atom IDs")
        positions = np.searchsorted(expected_ids, normalized)
        ordered_labels = self._labels[expected_order][positions]
        return GrainOwnership(
            atom_ids=normalized,
            labels=ordered_labels,
            gb_plane_x=self.gb_plane_x,
            inplane_periodic=self.inplane_periodic,
            left_grain_x_bounds=self._left_grain_x_bounds,
            right_grain_x_bounds=self._right_grain_x_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            normal_topology=self._normal_topology,
        )

    def __copy__(self) -> GrainOwnership:
        """Independent immutable ownership copy."""
        # doccheck: ignore=DOC115[GrainOwnershipError]
        #   Frozen/read-only state cannot become invalid through the supported API;
        #   constructor failure here would indicate unsupported internal corruption.
        return GrainOwnership(
            atom_ids=self._atom_ids,
            labels=self._labels,
            gb_plane_x=self.gb_plane_x,
            inplane_periodic=self.inplane_periodic,
            left_grain_x_bounds=self._left_grain_x_bounds,
            right_grain_x_bounds=self._right_grain_x_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            normal_topology=self._normal_topology,
        )

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> GrainOwnership:
        """Return an independent immutable ownership deep copy.

        :param memo: Standard deep-copy memo used to retain identity mappings.
        :return: New ownership object containing copied arrays and identical metadata.
        """
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied
