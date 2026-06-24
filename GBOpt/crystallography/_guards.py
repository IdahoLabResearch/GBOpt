# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared precondition guards for crystallography functions.

Contains lightweight guard functions that enforce package-wide preconditions at public
entry points. Guards raise a consistent exception from the crystallography exception
hierarchy rather than letting invalid inputs propagate silently into arithmetic.

Each guard is called at the top of any function that accepts the corresponding
parameter. Adding a new guard here rather than inline in individual modules ensures that
the error message, exception type, and condition are defined and tested exactly once.

Currently implemented guards:

- ``_require_cubic``: rejects non-``None`` lattice metrics until non-cubic support is
  implemented.

This module is private to the crystallography package and should not be imported
directly by external callers.
"""

import numpy as np

from .types import CrystallographyNotImplementedError


def _require_cubic(metric: np.ndarray | None) -> None:
    """Raise if a non-cubic lattice metric is supplied.

    Exact CSL support is currently implemented only for the implicit cubic identity
    metric. This guard is called at the entry point of any function that accepts a
    ``lattice_metric`` parameter to give a consistent error until non-cubic support is
    implemented.

    .. note::
        This function will be removed once non-cubic lattice support is implemented. At that
        point, callers should replace this guard with real metric handling.

    :param metric: Lattice metric tensor to check; ``None`` is the only accepted value.
    :raises CrystallographyNotImplementedError: If ``metric`` is not ``None``.
    """
    if metric is not None:
        raise CrystallographyNotImplementedError(
            "non-cubic lattice metrics are not implemented"
        )
