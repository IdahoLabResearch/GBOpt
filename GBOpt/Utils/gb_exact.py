# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact-solver utilities for canonical P/Q bicrystal construction.

Functions here are filled in incrementally across Stages B and C.
Each stub raises NotImplementedError until its owning step is complete.
"""

import numpy as np

from GBOpt.BoundarySpec import BoundarySpecError


def validate_and_normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Validate that quat is an integer quaternion and return its normalized form.

    Parameters
    ----------
    quat : array-like of shape (4,)
        Candidate integer quaternion [a, b, c, d].

    Returns
    -------
    np.ndarray of shape (4,) with dtype float
        Normalized quaternion (unit length).

    Raises
    ------
    BoundarySpecError
        If any component is non-integer or the quaternion is zero.
    """
    raise NotImplementedError


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a normalized integer quaternion to an exact rotation matrix.

    Parameters
    ----------
    quat : np.ndarray of shape (4,)
        Normalized unit quaternion [a, b, c, d] (integer-origin).

    Returns
    -------
    np.ndarray of shape (3, 3)
        Exact rotation matrix corresponding to quat.
    """
    raise NotImplementedError


def validate_sigma(quat: np.ndarray, sigma: int) -> None:
    """Validate that sigma derived from quat matches the user-supplied value.

    Sigma is derived from the integer quaternion as the norm-squared divided
    by its largest power-of-2 factor, which is always an exact odd integer.
    Validation is therefore an exact equality check with no tolerance.

    Parameters
    ----------
    quat : np.ndarray of shape (4,)
        Integer quaternion (unnormalized).
    sigma : int
        User-provided sigma value to validate against.

    Raises
    ------
    BoundarySpecError
        If the derived sigma does not exactly match the provided value.
    """
    raise NotImplementedError


def solve_inplane_csl(
    axis: np.ndarray,
    plane: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the exact in-plane CSL basis from rotation axis, boundary plane, and R.

    Parameters
    ----------
    axis : np.ndarray of shape (3,)
        Integer Miller rotation axis [u v w].
    plane : np.ndarray of shape (3,)
        Integer Miller boundary-plane normal [h k l].
    R : np.ndarray of shape (3, 3)
        Exact rotation matrix from quaternion_to_rotation_matrix.

    Returns
    -------
    v1, v2 : np.ndarray of shape (3,)
        Two in-plane CSL basis vectors (pre-reduction).

    Raises
    ------
    BoundarySpecError
        If no exact CSL reconstruction is found or the input is not a
        rational CSL boundary.
    """
    raise NotImplementedError


def reduce_2d_basis(
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply 2D lattice reduction to a pair of in-plane basis vectors.

    Parameters
    ----------
    v1, v2 : np.ndarray of shape (3,)
        In-plane basis vectors to reduce.

    Returns
    -------
    r1, r2 : np.ndarray of shape (3,)
        Reduced basis vectors in canonical (short, deterministic) form.
    """
    raise NotImplementedError


def canonicalize_pq(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical forms of the P and Q orientation matrices.

    Canonicalization rules:
    - rows are integer or exact rational directions reduced by gcd
    - matrices are right-handed (positive determinant after normalization)
    - row 0 is the boundary normal
    - rows 1-2 form a deterministic reduced in-plane basis
    - sign and ordering conventions are fixed so equivalent inputs
      canonicalize identically

    Parameters
    ----------
    P, Q : np.ndarray of shape (3, 3)
        Row-wise orientation matrices for each grain.

    Returns
    -------
    P_canon, Q_canon : np.ndarray of shape (3, 3)
        Canonicalized orientation matrices.
    """
    raise NotImplementedError


def exactify_five_dof(
    params: np.ndarray,
    max_exact_atoms: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Attempt to rationalize a 5-DOF input to exact canonical P/Q.

    Bounded search from floating 5-DOF [alpha, beta, gamma, theta, phi]
    to the nearest cubic CSL boundary. Deferred — raises NotImplementedError
    until Stage E.

    Parameters
    ----------
    params : np.ndarray of shape (5,)
        Five-DOF misorientation parameters in radians.
    max_exact_atoms : int
        Upper bound on commensurate cell size; raises BoundarySpecError
        if the exact cell would exceed this limit.

    Returns
    -------
    P_canon, Q_canon : np.ndarray of shape (3, 3)
        Canonical orientation matrices for the nearest CSL boundary.

    Raises
    ------
    NotImplementedError
        Always, until Stage E is implemented.
    BoundarySpecError
        If no CSL boundary is found within max_exact_atoms.
    """
    raise NotImplementedError
