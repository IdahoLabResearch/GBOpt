# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define physical topology along the grain-boundary normal.

This module owns the immutable topology vocabulary shared by construction and
interface manipulation. Coordinate inference, external file parsing, atom generation,
and optimizer policy do not belong here.
"""

from __future__ import annotations

from enum import Enum


class BoundaryTopologyError(ValueError):
    """Raised when boundary-normal topology metadata is invalid."""


class BoundaryNormalTopology(str, Enum):
    """Physical topology along the grain-boundary normal.

    ``PERIODIC_BICRYSTAL``
        The central grain boundary is accompanied by a second interface across the
        periodic outer x faces.

    ``SINGLE_INTERFACE_SLAB``
        The structure contains one central grain boundary and outer vacuum/free-surface
        intervals rather than a second periodic interface.

    ``UNKNOWN``
        Available metadata is insufficient to establish either topology.
    """

    PERIODIC_BICRYSTAL = "periodic_bicrystal"
    SINGLE_INTERFACE_SLAB = "single_interface_slab"
    UNKNOWN = "unknown"

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Return whether the outer x faces form a second physical interface.

        :return: ``True`` only for a periodic bicrystal.
        """
        return self is BoundaryNormalTopology.PERIODIC_BICRYSTAL


def normalize_boundary_normal_topology(
    value: BoundaryNormalTopology | str | None,
) -> BoundaryNormalTopology:
    """Return validated boundary-normal topology metadata.

    :param value: Topology as an enum member, serialized string, or ``None`` when
        topology metadata is unavailable.
    :return: Validated topology. ``None`` normalizes to ``UNKNOWN``.
    :raises BoundaryTopologyError: If ``value`` is not supported.
    """
    if value is None:
        return BoundaryNormalTopology.UNKNOWN

    try:
        return BoundaryNormalTopology(value)
    except (TypeError, ValueError) as exc:
        raise BoundaryTopologyError(
            f"Unsupported boundary-normal topology: {value!r}"
        ) from exc


__all__ = [
    "BoundaryNormalTopology",
    "BoundaryTopologyError",
    "normalize_boundary_normal_topology",
]
