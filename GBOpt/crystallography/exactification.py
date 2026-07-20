# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Rationalization of approximate boundaries into exact crystallographic forms."""

from __future__ import annotations

import numpy as np


def exactify_five_dof(
    params: np.ndarray,
    *,
    max_exact_atoms: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert five-DOF boundary parameters to exact canonical P/Q matrices.

    :param params: Five-DOF parameters ``[alpha, beta, gamma, theta, phi]`` in radians.
    :param max_exact_atoms: Maximum permitted size of the resulting exact cell.
        Keyword argument, optional, defaults to ``10000``.
    :return: Canonical integer orientation matrices ``(P, Q)``.
    :raises NotImplementedError: Five-DOF exactification is not yet implemented.
    """
    raise NotImplementedError(
        "Conversion from five-DOF parameters to exact canonical P/Q matrices "
        "is not implemented."
    )


__all__ = ["exactify_five_dof"]
