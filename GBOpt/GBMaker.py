# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Grain boundary builder utilities."""

import math
import warnings
from numbers import Number
from typing import Any, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from GBOpt.UnitCell import UnitCell
from GBOpt.BoundarySpec import BoundarySpecError, CSLApproxSpec, CSLExactSpec, PQSpec
from GBOpt.Utils.gb_exact import (
    csl_approx_spec_to_embedding,
    csl_spec_to_embedding,
    pq_spec_to_embedding,
)


def _find_commensurate_pair(
    d1: float,
    d2: float,
    tol: float,
    max_n: int,
) -> "tuple[int, int, float, float] | None":
    """Find the smallest integer pair (n1, n2) such that n1*d1 ≈ n2*d2.

    Searches all pairs with 1 ≤ n1, n2 ≤ max_n and returns the one with
    the smallest ``max(n1*d1, n2*d2)`` whose relative mismatch satisfies::

        |n1*d1 - n2*d2| / max(n1*d1, n2*d2) ≤ tol

    Returns ``(n1, n2, n1*d1, n2*d2)`` or ``None`` if no pair is found.

    :param d1: Period of the first grain (Angstroms).
    :param d2: Period of the second grain (Angstroms).
    :param tol: Maximum allowed relative mismatch (e.g. ``0.005`` for 0.5%).
    :param max_n: Upper bound on n1 and n2.
    :return: Best commensurate pair or ``None``.

    .. todo::
        For non-cubic structures with oblique in-plane cells, the period
        lengths depend on the full metric tensor and the two in-plane axes
        may need to be treated jointly when the cell is not rectangular.
    """
    best: "tuple[int, int, float, float] | None" = None
    best_size = float("inf")
    for n1 in range(1, max_n + 1):
        l1 = n1 * d1
        for n2 in range(1, max_n + 1):
            l2 = n2 * d2
            mismatch = abs(l1 - l2) / max(l1, l2)
            if mismatch <= tol and max(l1, l2) < best_size:
                best = (n1, n2, l1, l2)
                best_size = max(l1, l2)
    return best


class GBMakerError(Exception):
    """Base class for Exceptions in the GBMaker class."""
    pass


class GBMakerTypeError(GBMakerError):
    """Exception raised when an invalid type is assigned to a GBMaker attribute."""
    pass


class GBMakerValueError(GBMakerError):
    """Exception raised when an invalid value is assigned to a GBMaker attribute."""
    pass


def wrap_reduced_coordinate(reduced_coord: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Wrap reduced coordinates into [0, 1) and snap both periodic faces to 0.

    :param reduced_coord: Reduced coordinates to wrap.
    :param tol: Tolerance in reduced-coordinate units. Optional, defaults to 1e-8
    :return: Wrapped reduced coordinates in [0, 1).
    """
    if not math.isfinite(tol):
        raise GBMakerValueError("Reduced-coordinate tolerance must be finite.")
    if tol < 0:
        raise GBMakerValueError("Reduced-coordinate tolerance must be non-negative.")

    wrapped = np.mod(np.asarray(reduced_coord, dtype=np.float64), 1.0)
    return np.where(
        (wrapped < tol) | ((1.0 - wrapped) < tol),
        0.0,
        wrapped,
    )


class GBMaker:
    """
    Class to create a GB structure based on user defined parameters. The GB normal is
    aligned along the x-axis.

    :param a0: Crystal lattice parameter (Angstroms).
    :param structure: Crystal structure string ['fcc', 'bcc', 'sc', 'diamond',
        'fluorite', 'rocksalt', 'zincblende'].
    :param gb_thickness: The width of the GB region (Angstroms).
    :param misorientation: Misorientation angles (alpha, beta, gamma, theta, phi) in
        radians. Alpha, beta, and gamma represent the ZXZ Euler angles, and theta and
        phi represent the additional rotation about y and z, respectively.
    :param repeat_factor: The number of times to repeat the unit cell in the y and z
        directions, optional, defaults to 2. A single integer value is assumed to be
        used for both directions, while a list of length 2 is assumed to apply
        sequentially to y and z. Values less than 2 give a warning.
    :param x_dim_min: Minimum size of one grain in the x dimension (Angstroms),
        optional, defaults to 50.
    :param vacuum: Thickness of the vacuum region around the grains in the x dimension
        (Angstroms), optional, defaults to 10.
    :param interaction_distance: The maximum distance that atoms interact with each
        other, optional, defaults to 15.0. If y_dim or z_dim are less than twice this
        number with the given repeat_factor(s), a new value is calculated that
        accommodates this value.
    :param gb_id: The identifier for the created GB system, optional, defaults to 0.
    """

    def __init__(self, a0: float, structure: str, gb_thickness: float,
                 misorientation: np.ndarray, atom_types: str | Tuple[str, ...], *,
                 _embedding=None,
                 _mismatch_tol=None,
                 _mismatch_max_cells: int = 50,
                 _strain_grain: str = "both",
                 repeat_factor: Union[int, Sequence[int]] = 2, x_dim_min: float = 50,
                 vacuum: float = 10, interaction_distance: float = 15.0,
                 gb_id: int = 1, epsilon: float = 1e-10):
        self.__a0 = self.__validate(a0, Number, "a0", positive=True)
        self.__structure = self.__validate(structure, str, "structure")
        self.__gb_thickness = self.__validate(
            gb_thickness, Number, "gb_thickness", positive=True
        )
        self.__epsilon = self.__validate(
            epsilon, Number, "epsilon", strictly_positive=True)
        self.__assign_orientations(
            self.__validate(
                np.asarray(misorientation),
                np.ndarray,
                "misorientation",
                expected_length=5,
            )
        )
        self.__repeat_factor = self.__validate(
            repeat_factor,
            (int, Sequence),
            "repeat_factor",
            expected_length=2,
            positive=True,
        )
        self.__x_dim_min = self.__validate(
            x_dim_min, Number, "x_dim_min", positive=True)
        self.__vacuum_thickness = self.__validate(
            vacuum, Number, "vacuum_thickness", positive=True
        )
        self.__interaction_distance = self.__validate(
            interaction_distance, Number, "interaction_distance", positive=True
        )
        self.__id = self.__validate(gb_id, int, "id", positive=True)
        self.__inplane_periodic = (True, True)
        self.__embedding = _embedding
        self.__mismatch_tol = _mismatch_tol
        self.__mismatch_max_cells = int(_mismatch_max_cells)
        self.__strain_grain = _strain_grain

        self.__unit_cell = self.__init_unit_cell(atom_types)
        self.__spacing = self.__calculate_periodic_spacing()  # periodic distances dict
        self.__update_dims()

        self.__radius = a0 * self.__unit_cell.radius  # atom radius
        self.__box_dims = self.__calculate_box_dimensions()

    @classmethod
    def _from_boundary_embedding(
        cls,
        embedding,
        *,
        a0: float,
        structure: str,
        atom_types,
        gb_thickness: float = 0.0,
        repeat_factor=2,
        x_dim_min: float = 50,
        vacuum: float = 10,
        interaction_distance: float = 15.0,
        gb_id: int = 1,
        mismatch_tol=None,
        mismatch_max_cells: int = 50,
        strain_grain: str = "both",
    ) -> "GBMaker":
        """Build a GBMaker from a BoundaryEmbedding.

        :param embedding: A BoundaryEmbedding produced by an input adapter. When
            ``embedding.exact`` is True and P/Q are present, the integer matrices are
            used directly as the approx rotation matrices, bypassing
            ``__approximate_rotation_matrix_as_int``. When ``embedding.exact`` is False,
            R_left/R_right are used on the existing floating-point approximation path.
            ``embedding.coherent`` sets ``inplane_periodic``.
        :param a0: Crystal lattice parameter (Angstroms).
        :param structure: Crystal structure string.
        :param atom_types: Atom type string or tuple of strings.
        :param gb_thickness: Width of the GB region (Angstroms), default 0.
        :param repeat_factor: In-plane repeat factor(s), default 2.
        :param x_dim_min: Minimum grain thickness in x (Angstroms), default 50.
        :param vacuum: Vacuum thickness (Angstroms), default 10.
        :param interaction_distance: Maximum atom interaction distance, default 15.
        :param gb_id: Grain boundary identifier, default 1.
        :param mismatch_tol: Passed through to ``_mismatch_tol``.
        :param mismatch_max_cells: Passed through to ``_mismatch_max_cells``.
        :param strain_grain: Passed through to ``_strain_grain``.
        :return: Fully initialized GBMaker instance.
        """
        return cls(
            a0, structure, gb_thickness, np.zeros(5), atom_types,
            _embedding=embedding,
            _mismatch_tol=mismatch_tol,
            _mismatch_max_cells=mismatch_max_cells,
            _strain_grain=strain_grain,
            repeat_factor=repeat_factor,
            x_dim_min=x_dim_min,
            vacuum=vacuum,
            interaction_distance=interaction_distance,
            gb_id=gb_id,
        )

    @classmethod
    def from_boundary_spec(
        cls,
        a0: float,
        structure: str,
        atom_types,
        boundary,
        mode: str = "exact",
        *,
        gb_thickness: float = 0.0,
        repeat_factor=2,
        x_dim_min: float = 50,
        vacuum: float = 10,
        interaction_distance: float = 15.0,
        gb_id: int = 1,
        mismatch_tol: "float | None" = None,
        mismatch_max_cells: int = 50,
        strain_grain: str = "both",
    ) -> "GBMaker":
        """Public factory that builds a GBMaker from a boundary-spec dataclass.

        Supported boundary types and modes:

        ============= =========== ===========  ============
        Boundary type exact        approximate  prefer_exact
        ============= =========== ===========  ============
        PQSpec        ✓            ✗            ✗
        CSLExactSpec  ✓            ✗            ✗
        CSLApproxSpec ✗ (raises)   ✓            ✗
        FiveDOFSpec   not yet      not yet      not yet
        ============= =========== ===========  ============

        **Commensurability requirement (exact mode only):**
        The shared in-plane simulation box is set to
        ``repeat_factor × max(grain1_period, grain2_period)`` along each
        in-plane axis.  For the exact path this box must be an integer
        multiple of *both* grains' in-plane periods; otherwise a
        ``GBMakerValueError`` is raised.  For symmetric CSL boundaries
        (e.g. Σ5 [001] tilt) the two grains share the same in-plane period
        so the condition is always satisfied.  For asymmetric or mixed
        tilt/twist boundaries where the two grains have incommensurate
        in-plane periods, use ``mode="approximate"`` or increase
        ``repeat_factor`` until the box is commensurate with both grains.
        A geometrically complete fix — constructing the shared box from the
        CSL in-plane cell rather than the maximum single-grain period — is
        not yet implemented.

        :param a0: Crystal lattice parameter (Angstroms).
        :param structure: Crystal structure string.
        :param atom_types: Atom type string or tuple of strings.
        :param boundary: A boundary-spec dataclass (``PQSpec``,
            ``CSLExactSpec``, or ``CSLApproxSpec``).
        :param mode: Construction mode — ``"exact"`` uses exact integer P/Q
            matrices (requires ``PQSpec`` or ``CSLExactSpec``);
            ``"approximate"`` uses floating-point rotation matrices
            (required for ``CSLApproxSpec``).
        :param gb_thickness: Width of the GB region (Angstroms), default 0.
        :param repeat_factor: In-plane repeat factor(s), default 2.
        :param x_dim_min: Minimum grain thickness in x (Angstroms), default 50.
        :param vacuum: Vacuum thickness (Angstroms), default 10.
        :param interaction_distance: Maximum atom interaction distance, default 15.
        :param gb_id: Grain boundary identifier, default 1.
        :param mismatch_tol: When not ``None``, search for integer multiples
            ``n1*d1 ≈ n2*d2`` (``approximate`` mode only) to minimize the
            in-plane mismatch strain.  Typical value: ``0.005`` (0.5%).
            Raises ``NotImplementedError`` for ``mode="exact"`` (exact path
            already enforces integer commensurability).
        :param mismatch_max_cells: Upper bound on n1 and n2 in the
            commensurability search, default 50.
        :param strain_grain: Which grain absorbs the residual mismatch.
            ``"both"`` (default) sets the box to the average of ``n1*d1``
            and ``n2*d2`` (symmetric strain); ``"left"`` uses ``n2*d2``
            (right grain exact); ``"right"`` uses ``n1*d1`` (left grain
            exact).  Ignored when ``mismatch_tol`` is ``None``.
        :return: Fully initialized GBMaker instance.
        :raises BoundarySpecError: If the mode is incompatible with the
            boundary type (e.g. ``CSLApproxSpec`` with ``mode="exact"``).
        :raises GBMakerValueError: If ``mode="exact"`` and the shared in-plane
            box is not commensurate with both grains' in-plane periods.
        :raises NotImplementedError: For unsupported boundary types, modes, or
            when ``mismatch_tol`` is combined with ``mode="exact"``.
        """
        if mismatch_tol is not None and mode == "exact":
            warnings.warn(
                "mismatch_tol has no effect with mode='exact': the exact path "
                "already enforces integer-commensurate in-plane periods via the "
                "membership kernel. mismatch_tol will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            mismatch_tol = None

        if isinstance(boundary, PQSpec):
            if mode != "exact":
                raise NotImplementedError(
                    f"Construction mode '{mode}' is not yet supported for PQSpec; "
                    f"only mode='exact' is currently implemented."
                )
            embedding = pq_spec_to_embedding(boundary)

        elif isinstance(boundary, CSLExactSpec):
            if mode != "exact":
                raise NotImplementedError(
                    f"Construction mode '{mode}' is not yet supported for CSLExactSpec; "
                    f"only mode='exact' is currently implemented."
                )
            embedding = csl_spec_to_embedding(boundary)

        elif isinstance(boundary, CSLApproxSpec):
            if mode == "exact":
                raise BoundarySpecError(
                    "CSLApproxSpec cannot be used with mode='exact': no integer "
                    "quaternion is available for exactification. Use CSLExactSpec "
                    "for an exact construction, or mode='approximate'."
                )
            if mode != "approximate":
                raise NotImplementedError(
                    f"Construction mode '{mode}' is not yet supported for CSLApproxSpec."
                )
            embedding = csl_approx_spec_to_embedding(boundary)

        else:
            raise NotImplementedError(
                f"from_boundary_spec does not yet support {type(boundary).__name__}."
            )
        return cls._from_boundary_embedding(
            embedding,
            a0=a0,
            structure=structure,
            atom_types=atom_types,
            gb_thickness=gb_thickness,
            repeat_factor=repeat_factor,
            x_dim_min=x_dim_min,
            vacuum=vacuum,
            interaction_distance=interaction_distance,
            gb_id=gb_id,
            mismatch_tol=mismatch_tol,
            mismatch_max_cells=mismatch_max_cells,
            strain_grain=strain_grain,
        )

    @staticmethod
    def __reduce_integer_row(row: np.ndarray) -> np.ndarray:
        """
        Reduce an integer row by its GCD

        :param row: Integer row vector
        :return: GCD-reduced integer row vector
        """
        reduced = np.asarray(row, dtype=int).copy()
        non_zero = np.abs(reduced[reduced != 0])
        if not non_zero.size:
            return reduced
        gcd = np.gcd.reduce(non_zero)
        if gcd > 1:
            reduced //= gcd
        return reduced

    @staticmethod
    def __row_angle_error_deg(reference: np.ndarray, candidate: np.ndarray) -> float:
        """
        Compute the angular error in degrees between two vectors.

        :param reference: Reference float vector.
        :param candidate: Candidate integer vector.
        :return: Angle between the two vectors in degrees
        """
        ref_norm = np.linalg.norm(reference)
        cand_norm = np.linalg.norm(candidate)
        if np.isclose(ref_norm, 0) or np.isclose(cand_norm, 0):
            return 180.0
        cosine = np.dot(reference, candidate) / (ref_norm * cand_norm)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    # Private class methods
    def __approximate_rotation_row_as_int(
        self, row: np.ndarray, angle_tol_deg: float = 0.5, max_scale: int = 10000
    ) -> np.ndarray:
        """
        Find the smallest-norm integer vector that points within *angle_tol_deg* of
        *row*. LLL lattice reduction was considered but not adopted: for CSL boundaries
        the correct scale factor k equals ||g|| <= √Sigma (typically < 10), so the loop exits
        in a handful of iterations. The LLL selection step also requires enumerating
        signed combinations of reduced basis rows — adding ~60 lines for no practical
        gain.

        :param row: A single float row from a rotation matrix.
        :param angle_tol_deg: Maximum allowed angular error in degrees.
        :param max_scale: Upper bound of the scale factor to try.
        :return: Best-matching integer vector.
        """
        row = np.asarray(row, dtype=np.float64)
        best = None
        best_err = 180.0
        batch_size = 1000
        for k_start in range(1, max_scale + 1, batch_size):
            k_end = min(k_start + batch_size, max_scale + 1)
            for k in range(k_start, k_end):
                candidate = self.__reduce_integer_row(
                    np.round(row * k).astype(int))
                err = self.__row_angle_error_deg(row, candidate)
                if err < best_err or (err == best_err and np.linalg.norm(candidate) < np.linalg.norm(best)):
                    best_err = err
                    best = candidate
                if best_err <= angle_tol_deg:
                    break
            if best_err <= angle_tol_deg:
                break
        return best if best is not None else np.round(row).astype(int)

    def __approximate_rotation_matrix_as_int(
        self, m: np.ndarray, precision: float = 5
    ) -> np.ndarray:
        """
        Approximate a rotation matrix in integer format given the original matrix and
        the desired precision.

        :param m: The matrix to approximate
        :param precision: Decimal precision to use during calculations, defaults to 5
        :return: Integer approximation of the rotation matrix m
        """

        max_scale = max(1000, 10**max(int(precision)-1, 0))
        return np.vstack(
            [
                self.__approximate_rotation_row_as_int(
                    row, angle_tol_deg=0.5, max_scale=max_scale
                )
                for row in np.asarray(m, dtype=np.float64)
            ]
        ).astype(int)

    def __assign_orientations(self, misorientation: np.ndarray) -> None:
        """
        Private method to separate the misorientation and inclination from the passed in
        misorientation array.

        :param misorientation: Array containing the misorientation and inclination Euler
            angles. Misorientation is the first three, and inclination is the last two.
            Note that misorientation is in the ZXZ Euler angle format.
        """
        self.__misorientation = misorientation[:3]
        self.__inclination = misorientation[3:]
        self.__Rmis = Rotation.from_euler(
            "ZXZ", misorientation[:3]).as_matrix()
        self.__Rincl = (
            Rotation.from_euler("z", misorientation[4])
            * Rotation.from_euler("y", misorientation[3])
        ).as_matrix()

    def __calculate_box_dimensions(self) -> np.ndarray:
        """
        Private method to calculate the box dimensions

        :return: The 3x2 array containing xlo, xhi, ylo, yhi, zlo, and zi.
        """
        return np.array(
            [
                [0, self.__x_dim + 2 * self.__vacuum_thickness],
                [0, self.__y_dim],
                [0, self.__z_dim],
            ]
        )

    def __exact_grain_repeats(
        self,
        P_or_Q: np.ndarray,
        x_length: float,
    ) -> tuple[int, int, int]:
        """Compute integer repeat counts (repeat_x, repeat_y, repeat_z) for an exact grain.

        Derives how many times the supercell period fits into the box along each
        axis.  The x count is derived from the already-equalized grain thickness
        ``x_length``.  The y and z counts are derived from ``self.__y_dim`` /
        ``self.__z_dim``.

        The commensurability guard raises if y or z does not divide evenly
        (within tolerance), which means the shared in-plane box is not an
        integer multiple of this grain's in-plane periods.  See the
        commensurability note in ``from_boundary_spec`` for guidance.

        :param P_or_Q: 3×3 canonical integer orientation matrix for this grain.
        :param x_length: Equalized x-slab thickness for this grain (Angstroms).
        :return: (repeat_x, repeat_y, repeat_z) as positive integers.
        :raises GBMakerValueError: If y or z is not commensurate with the box.
        """
        from GBOpt.Utils.gb_exact import build_supercell_matrix

        S = build_supercell_matrix(P_or_Q)
        a0 = self.__a0

        x_period = a0 * float(np.linalg.norm(S[0]))
        y_period = a0 * float(np.linalg.norm(S[1]))
        z_period = a0 * float(np.linalg.norm(S[2]))

        repeat_x = int(round(x_length / x_period))
        repeat_y_raw = self.__y_dim / y_period
        repeat_z_raw = self.__z_dim / z_period

        repeat_y = int(round(repeat_y_raw))
        repeat_z = int(round(repeat_z_raw))

        tol = 1e-6
        if abs(repeat_y_raw - repeat_y) > tol:
            raise GBMakerValueError(
                f"Exact construction requires the y box ({self.__y_dim:.6f} Å) to be "
                f"an integer multiple of this grain's y-period ({y_period:.6f} Å), "
                f"but got repeat_y = {repeat_y_raw:.8f}.  "
                "Use mode='approximate' or adjust repeat_factor until both grains' "
                "in-plane periods divide the shared box exactly. See the "
                "commensurability note in from_boundary_spec for details."
            )
        if abs(repeat_z_raw - repeat_z) > tol:
            raise GBMakerValueError(
                f"Exact construction requires the z box ({self.__z_dim:.6f} Å) to be "
                f"an integer multiple of this grain's z-period ({z_period:.6f} Å), "
                f"but got repeat_z = {repeat_z_raw:.8f}.  "
                "Use mode='approximate' or adjust repeat_factor until both grains' "
                "in-plane periods divide the shared box exactly. See the "
                "commensurability note in from_boundary_spec for details."
            )

        return repeat_x, repeat_y, repeat_z

    def __generate_grain_exact(
        self,
        R_grain: np.ndarray,
        P_or_Q: np.ndarray,
        x_length: float,
        x_offset: float,
    ) -> np.ndarray:
        """Build one grain using the integer membership kernel (exact path).

        Enumerates conventional-cell origins via pure-integer arithmetic,
        expands each accepted origin to the full ``UnitCell.asarray()`` basis,
        rotates to the lab frame, and translates by ``x_offset``.  The float
        selection/clipping/deduplication routines (``__select_atoms_in_box_basis``,
        ``__clip_atoms_to_cartesian_box``, ``__deduplicate_positions``) and
        ``trim_upper_face`` are **never called** on this path.

        :param R_grain: Proper rotation matrix for this grain.
        :param P_or_Q: 3×3 canonical integer orientation matrix.
        :param x_length: Equalized x-slab thickness (Angstroms).
        :param x_offset: Lab x-coordinate of the grain's lower face (Angstroms).
        :return: Structured atom array in the same format as ``__generate_grain``.
        """
        from GBOpt.Utils.gb_exact import build_supercell_matrix, enumerate_supercell_origins

        repeat_x, repeat_y, repeat_z = self.__exact_grain_repeats(P_or_Q, x_length)
        S = build_supercell_matrix(P_or_Q)
        origins = enumerate_supercell_origins(S, repeat_x, repeat_y, repeat_z)

        # Cartesian origins in the crystal frame: n @ (a0 * I) = a0 * n
        cart_origins = origins.astype(np.float64) * self.__a0

        atoms = self.get_supercell(cart_origins)

        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        rotated = positions @ R_grain.T
        rotated[:, 0] += x_offset
        atoms["x"], atoms["y"], atoms["z"] = rotated.T

        # Invariant: total atoms = accepted origins × basis size
        uc_size = len(self.__unit_cell.asarray())
        assert len(atoms) == len(origins) * uc_size, (
            f"Exact builder atom count mismatch: expected "
            f"{len(origins) * uc_size}, got {len(atoms)}"
        )
        return atoms

    def __generate_gb(self) -> None:
        """
        Private method to calculate the left grain, right grain, and whole GB system
        """
        left_bounds = np.array(
            [
                self.__vacuum_thickness,
                self.__left_x + self.__vacuum_thickness,
            ],
            dtype=np.float64,
        )
        right_bounds = np.array(
            [
                self.__left_x + self.__vacuum_thickness,
                self.__x_dim + self.__vacuum_thickness,
            ],
            dtype=np.float64,
        )

        use_exact = (
            self.__embedding is not None
            and self.__embedding.exact
            and self.__embedding.coherent
            and self.__embedding.P is not None
        )

        if use_exact:
            self.__left_grain = self.__generate_grain_exact(
                self.__R_left,
                self.__embedding.P,
                self.__left_x,
                left_bounds[0],
            )
            self.__right_grain = self.__generate_grain_exact(
                self.__R_right,
                self.__embedding.Q,
                self.__right_x,
                right_bounds[0],
            )
        else:
            self.__left_grain = self.__generate_grain(
                self.__R_left,
                self.__R_left_approx,
                left_bounds,
            )
            # For vacuum=0 (periodic-in-x), the right grain's last x-period
            # would wrap onto the left grain's first plane under PBC, creating
            # a spurious double interface. Build one period shorter so the
            # grain ends before its own upper boundary — stoichiometric because
            # we exclude a complete origin period rather than float-slicing.
            # Guard: only trim when at least two x-periods remain after the
            # reduction. If the grain spans only one period the trim would
            # remove all atoms; in that case skip the stoichiometric trim and
            # rely on the gap-equalization step below.
            x_period_right = self.__x_period(self.__R_right_approx)
            right_bounds_eff = right_bounds.copy()
            if (self.__vacuum_thickness == 0
                    and right_bounds_eff[1] - right_bounds_eff[0]
                    > x_period_right * (1 + self.__epsilon)):
                right_bounds_eff[1] -= x_period_right
            self.__right_grain = self.__generate_grain(
                self.__R_right,
                self.__R_right_approx,
                right_bounds_eff,
            )

        # The two grains may have different interplanar x-spacings (asymmetric tilt,
        # mixed tilt/twist, etc.), causing the periodic-edge gap to differ from the
        # central GB gap. When the periodic gap is smaller, close-contact atom pairs
        # form across the x boundary. Extend the trim to equalize the two gaps.
        left_max_x = np.max(self.__left_grain["x"])
        right_min_x = np.min(self.__right_grain["x"])
        central_gap = right_min_x - left_max_x
        left_min_x = np.min(self.__left_grain["x"])
        right_max_x = np.max(self.__right_grain["x"])
        periodic_gap = ((right_bounds[1] - right_max_x) +
                        (left_min_x - left_bounds[0]))
        if periodic_gap < central_gap - self.__epsilon:
            if use_exact:
                # Exact grains are filled by whole unit-cell periods; a float
                # x-threshold would cut partial crystal planes and destroy
                # stoichiometry. Unit-cell offsets routinely push a few atoms
                # just past the nominal grain boundary, making the periodic
                # gap slightly smaller than the central gap. Because the
                # bicrystal is bracketed by vacuum the atoms remain inside
                # the LAMMPS box and no close contacts form. Stoichiometry
                # takes precedence over exact gap symmetry on this path.
                pass
            else:
                # Period-based equalization: compute how many whole x-periods
                # of the right grain must be removed so that the periodic gap
                # grows to match the central gap.  Removing n complete periods
                # is stoichiometric (adjacent periods overlap in x-coordinate
                # space for multi-species structures, so no float threshold can
                # cleanly separate them — see Step 15a rationale).
                excess = right_max_x - (right_bounds[1] - central_gap)
                n_remove = max(1, math.ceil(excess / x_period_right))
                new_upper = right_bounds_eff[1] - n_remove * x_period_right
                if new_upper <= right_bounds_eff[0]:
                    warnings.warn(
                        f"Gap equalization would remove all atoms from the right "
                        f"grain ({n_remove} x-periods; right_x = "
                        f"{right_bounds_eff[1] - right_bounds_eff[0]:.4f} Å, "
                        f"x_period = {x_period_right:.4f} Å). "
                        "Skipping equalization to preserve non-empty grain."
                    )
                else:
                    if n_remove * x_period_right > (right_bounds_eff[1] - right_bounds_eff[0]) / 2:
                        warnings.warn(
                            f"Gap equalization removed {n_remove} x-period(s) "
                            f"({n_remove * x_period_right:.4f} Å) — more than half "
                            "the right grain. The resulting bicrystal may be unusable."
                        )
                    self.__right_grain = self.__generate_grain(
                        self.__R_right,
                        self.__R_right_approx,
                        np.array([right_bounds_eff[0], new_upper], dtype=np.float64),
                    )
            # When d_L < d_R (left grain denser in x), the periodic-edge gap equals d_R
            # which exceeds central_gap, so the fix above does not fire. This leaves
            # extra interfacial volume at the x periodic boundary. NPT simulations will
            # relax it away; NVT or static calculations will have structurally
            # inequivalent interfaces on the two sides of the bicrystal.

        self.__whole_system = np.hstack(
            (self.__left_grain, self.__right_grain))

    def __calculate_periodic_spacing(self, threshold: float = None) -> dict:
        """
        Calculate the periodic spacing based on the rotation matrix.

        :param threshold: The maximum allowed value that any spacing can take. Default
            is 15 * a0.
        :return: Dict containing the periodic spacing along the 'x', 'y', and 'z'
            directions for the given misorientation.
        """
        if threshold is None:
            threshold = self.__a0 * 15

        if self.__embedding is not None:
            # Exact or approximate path driven by a BoundaryEmbedding.
            self.__R_left = self.__embedding.R_left
            self.__R_right = self.__embedding.R_right
            if self.__embedding.exact and self.__embedding.P is not None:
                # Use exact integer matrices directly; bypass approximation.
                # Round first so float-valued canonical arrays (e.g. 4.0)
                # become Python ints, then cast to object dtype so that large
                # Miller indices don't overflow int64 in norm calculations.
                self.__R_left_approx = np.round(
                    self.__embedding.P).astype(int).astype(object)
                self.__R_right_approx = np.round(
                    self.__embedding.Q).astype(int).astype(object)
            else:
                self.__R_left_approx = self.__approximate_rotation_matrix_as_int(
                    self.__R_left).astype(object)
                self.__R_right_approx = self.__approximate_rotation_matrix_as_int(
                    self.__R_right).astype(object)
        else:
            # Legacy path: derive rotation matrices from Euler angles.
            self.__R_left = self.__Rincl
            self.__R_right = np.dot(self.__Rincl, self.__Rmis)
            # We store the approximate matrices as objects to allow for large numbers
            self.__R_left_approx = self.__approximate_rotation_matrix_as_int(
                self.__R_left).astype(object)
            self.__R_right_approx = self.__approximate_rotation_matrix_as_int(
                self.__R_right).astype(object)

        # The periodic distance in each direction is the lattice parameter multiplied by
        # norm of the Miller indices in that direction. This is determined using the
        # usual formula for the interplanar spacing: d = a / sqrt(h**2+k**2+l**2). The
        # square of the denominator here is the number of planes needed before
        # periodicity. Thus, if we multiply that distance by the interplanar spacing we
        # will get the interplanar spacing. This simplifies to
        # (a0**2/d**2)*d = a0**2/d --> spacing = a0 * sqrt(h**2+k**2+l**2)
        spacing_left = {
            axis: self.__a0 * np.linalg.norm(vec)
            for axis, vec in zip(["x", "y", "z"], self.__R_left_approx)
        }
        spacing_right = {
            axis: self.__a0 * np.linalg.norm(vec)
            for axis, vec in zip(["x", "y", "z"], self.__R_right_approx)
        }

        spacing = {
            "x": {"left": spacing_left["x"], "right": spacing_right["x"]}}
        self.__left_x = math.ceil(
            self.__x_dim_min / spacing["x"]["left"]) * spacing["x"]["left"]
        self.__right_x = math.ceil(
            self.__x_dim_min / spacing["x"]["right"]) * spacing["x"]["right"]
        target = max(self.__left_x, self.__right_x)
        self.__left_x = math.ceil(
            target / spacing["x"]["left"] - self.__epsilon) * spacing["x"]["left"]
        self.__right_x = math.ceil(
            target / spacing["x"]["right"] - self.__epsilon) * spacing["x"]["right"]
        self.__x_dim = self.__left_x + self.__right_x
        spacing.update(
            {
                axis: max(spacing_left[axis], spacing_right[axis])
                for axis in ["y", "z"]
            }
        )

        if self.__embedding is not None:
            # Trust the embedding's coherence flag directly; it is set by
            # the input adapter and reflects the exact construction intent.
            coherent = self.__embedding.coherent
            self.__inplane_periodic = (coherent, coherent)
        else:
            inplane_periodic = []
            for key, val in spacing.items():
                if key == 'x':
                    continue
                is_periodic = val <= threshold
                inplane_periodic.append(is_periodic)
                if not is_periodic:
                    spacing[key] = threshold
                    warnings.warn(
                        f"Required {key}-spacing {val:.4f} A exceeds threshold "
                        f"{threshold:.4f} A; boundary is non-periodic along {key}."
                    )
            self.__inplane_periodic = tuple(inplane_periodic)

        return spacing

    def __get_triclinic_params(self):
        """
        Computes the LAMMPS restricted-triclinic tilt factors. The y-period in the lab
        frame is R_grain @ (g_y * a0). For an exact CSL boundary this is exactly
        ||g_y|| * a0 * e_y; for non-CSL it has small x and z components. To satisfy
        LAMMPS's restriction that the b-vector lies in the xy-plane, rotate everything
        about the x-axis by theta = -atan2(A2[2], A2[1]).

        :return: (xy, xz, yz, theta) - the three tilt scalars and the rotation angle to
                                       apply to atom coordinates
        """
        if not all(self.__inplane_periodic):
            raise GBMakerValueError(
                "Triclinic output requires periodic y and z directions."
            )

        # Use grain with larger y-period, consistent with how spacing["y"] is chosen
        if np.linalg.norm(self.__R_left_approx[1]) >= np.linalg.norm(self.__R_right_approx[1]):
            R_grain = self.__R_left
            R_grain_approx = self.__R_left_approx
        else:
            R_grain = self.__R_right
            R_grain_approx = self.__R_right_approx

        # conventional stores basis vectors as rows: C = [a1; a2; a3].
        # Rotating each row vector to the lab frame gives [R@a1; R@a2; R@a3],
        # which in batch form is (R @ C.T).T = C @ R.T.
        rotated_unit_cell_basis = self.__unit_cell.conventional @ R_grain.T
        primitive_periods = (
            np.asarray(R_grain_approx[1:], dtype=np.float64) @ rotated_unit_cell_basis
        )
        A2_lab, A3_lab = self.__box_periodic_basis(primitive_periods)

        # Rotate about x to bring A2 into the xy-plane (LAMMPS restricted-triclinic
        # requires b-vector in the xy-plane). x-components are unaffected by this
        # rotation
        theta = -math.atan2(float(A2_lab[2]), float(A2_lab[1]))
        ct, st = math.cos(theta), math.sin(theta)

        # The x-rotation matrix is [[1,0,0],[0,ct,-st],[0,st,ct]]. The x-components of
        # A2_lab and A3_lab are unchanged by it, so xy and xz can be read direcly from
        # the pre-rotation vectors. yz requires the full rotation.
        xy = float(A2_lab[0])
        xz = float(A3_lab[0])
        yz = float(ct * A3_lab[1] - st * A3_lab[2])

        return xy, xz, yz, theta

    def __init_unit_cell(self, atom_types: str | Tuple[str, ...]) -> UnitCell:
        """
        Initializes the unit cell.

        :return: The unit cell initialized by structure.
        """
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.__structure, self.__a0, atom_types)
        return unit_cell

    def __x_period(self, R_grain_approx: np.ndarray) -> float:
        """Return one full x-period length for the given grain.

        The x-period is the distance between equivalent crystallographic
        planes along the boundary-normal direction, equal to
        ``a0 * ||R_grain_approx[0]||``.

        :param R_grain_approx: Integer approximation of the grain rotation matrix.
        :return: x-period in Angstroms.
        """
        return self.__a0 * float(
            np.linalg.norm(np.asarray(R_grain_approx[0], dtype=float))
        )

    def __generate_grain(
        self,
        R_grain: np.ndarray,
        R_grain_approx: np.ndarray,
        x_bounds: np.ndarray,
    ) -> np.ndarray:
        """
        Generate one grain by enumerating lattice coefficients over a bounded slab.

        :param R_grain: Rotation matrix for the grain.
        :param R_grain_approx: Integer approximation of ``R_grain``.
        :param x_bounds: Length-2 array-like containing ``[x_min, x_max]``.
        :return: Structured atom array for the selected grain.
        """
        x_bounds = np.asarray(x_bounds, dtype=np.float64)

        rotated_unit_cell_basis = self.__unit_cell.conventional @ R_grain.T
        primitive_periods = np.asarray(R_grain_approx[1:], dtype=np.float64)
        primitive_periods = primitive_periods @ rotated_unit_cell_basis

        reduced_periods = np.linalg.solve(
            rotated_unit_cell_basis.T, primitive_periods.T
        ).T
        x_direction_lattice = np.cross(reduced_periods[0], reduced_periods[1])
        rounded_direction = np.rint(x_direction_lattice)
        if np.allclose(
            x_direction_lattice, rounded_direction, atol=self.__epsilon, rtol=0.0
        ) and np.any(rounded_direction):
            x_direction_lattice = self.__reduce_integer_row(
                rounded_direction.astype(int)
            ).astype(np.float64)

        selection_box_basis = self.__selection_basis_vectors(primitive_periods).copy()
        axis_dims = (self.__y_dim, self.__z_dim)
        inplane_periodic = self.__inplane_periodic
        for row_index, (is_periodic, axis_dim) in enumerate(
            zip(inplane_periodic, axis_dims)
        ):
            if not is_periodic:
                selection_box_basis[row_index] *= axis_dim

        selection_box_basis_lattice = np.linalg.solve(
            rotated_unit_cell_basis.T, selection_box_basis.T
        ).T
        local_x_bounds = np.array([0.0, x_bounds[1] - x_bounds[0]], dtype=np.float64)
        nx_range = self.__x_index_range(
            primitive_periods, rotated_unit_cell_basis, local_x_bounds
        )

        lattice_bound_corners = []
        for nx in (nx_range[0], nx_range[-1]):
            x_base = nx * x_direction_lattice
            for uy in (0.0, 1.0):
                for uz in (0.0, 1.0):
                    cell_origin = (
                        x_base
                        + uy * selection_box_basis_lattice[0]
                        + uz * selection_box_basis_lattice[1]
                    )
                    for cell_corner in np.ndindex((2, 2, 2)):
                        lattice_bound_corners.append(
                            cell_origin + np.array(cell_corner, dtype=np.float64)
                        )

        lattice_bound_corners = np.asarray(lattice_bound_corners, dtype=np.float64)
        lattice_min = np.floor(np.min(lattice_bound_corners, axis=0)).astype(int) - 1
        lattice_max = np.ceil(np.max(lattice_bound_corners, axis=0)).astype(int) + 1

        coefficient_ranges = [
            np.arange(lower, upper + 1, dtype=int)
            for lower, upper in zip(lattice_min, lattice_max)
        ]
        lattice_coefficients = np.array(
            np.meshgrid(*coefficient_ranges, indexing="ij")
        ).reshape(3, -1).T

        atoms = self.get_supercell(lattice_coefficients @ self.__unit_cell.conventional)
        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        rotated_positions = positions @ R_grain.T
        rotated_positions[:, 0] += x_bounds[0]
        atoms["x"], atoms["y"], atoms["z"] = rotated_positions.T

        if any(inplane_periodic):
            atoms = self.__select_atoms_in_box_basis(atoms, primitive_periods, x_bounds)
        atoms = self.__clip_atoms_to_cartesian_box(atoms, x_bounds)

        return self.__deduplicate_positions(atoms)

    def __set_gb_region(self):
        """
        Identifies the atoms in the GB region based on the gb thickness.
        """
        x_gb = self.__vacuum_thickness + self.__left_x
        left_cut = x_gb - self.__gb_thickness / 2.0
        right_cut = x_gb + self.__gb_thickness / 2.0
        left_gb = self.__left_grain[self.__left_grain['x'] > left_cut]
        right_gb = self.__right_grain[self.__right_grain['x'] < right_cut]
        self.__gb_region = np.hstack((left_gb, right_gb))

    def __reduced_coordinate_tolerance(self, basis_vector: np.ndarray) -> float:
        """
        Convert the Cartesian epsilon to reduced-coordinate units for a basis vector.

        :param basis_vector: Cartesian basis vector used to define the coordinate scale.
        :return: Reduced-coordinate tolerance corresponding to ``self.__epsilon``.
        """
        basis_vector = np.asarray(basis_vector, dtype=np.float64)
        basis_length = np.linalg.norm(basis_vector)

        return self.__epsilon / basis_length

    def __scaled_periodic_basis_vector(
        self, period_vector: np.ndarray, box_length: float, axis_index: int
    ) -> np.ndarray:
        """
        Scale a periodic basis vector so one axis projection matches the box length.

        :param period_vector: Cartesian periodic basis vector.
        :param box_length: Desired box length along the selected axis.
        :param axis_index: Axis whose projection should match ``box_length``.
        :return: Scaled periodic basis vector.
        """

        period_vector = np.asarray(period_vector, dtype=np.float64)
        box_length = float(box_length)
        if box_length <= 0.0:
            raise GBMakerValueError("box_length must be strictly positive.")
        axis_index = int(axis_index)

        # We ignore overflow/invalid values because the check immediately after catches
        # those states and raises a GBMakerValueError
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            scale = box_length / period_vector[axis_index]
            scaled_vector = period_vector * scale
        if not np.all(np.isfinite(scaled_vector)):
            raise GBMakerValueError("Scaled periodic basis vector must be finite.")
        return scaled_vector

    def __box_periodic_basis(self, primitive_periods: np.ndarray) -> np.ndarray:
        """
        Build the in-plane box basis from primitive periodic vectors.

        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :return: 2x3 array containing the box basis vectors for y and z.
        """
        primitive_periods = np.asarray(primitive_periods, dtype=np.float64)

        inplane_periodic = self.__inplane_periodic
        box_lengths = (self.__y_dim, self.__z_dim)
        box_basis = np.zeros((2, 3), dtype=np.float64)

        for row_index, (is_periodic, box_length) in enumerate(
            zip(inplane_periodic, box_lengths)
        ):
            if not is_periodic:
                continue

            axis_index = row_index + 1
            axis_projection = primitive_periods[row_index, axis_index]
            if np.isclose(axis_projection, 0.0, atol=self.__epsilon, rtol=0.0):
                raise GBMakerValueError(
                    "primitive_periods must have a non-zero projection on the "
                    "selected box axis."
                )
            box_basis[row_index] = self.__scaled_periodic_basis_vector(
                primitive_periods[row_index], box_length, axis_index
            )

        return box_basis

    def __selection_basis_vectors(self, primitive_periods: np.ndarray) -> np.ndarray:
        """
        Build the canonical in-plane selection basis for y/z box coordinates.

        Periodic axes use the box-periodic basis vectors; non-periodic axes fall back
        to the corresponding Cartesian unit vectors.

        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :return: 2x3 array containing the y/z selection basis vectors.
        """
        selection_basis = self.__box_periodic_basis(primitive_periods)
        inplane_periodic = self.__inplane_periodic

        for row_index, is_periodic in enumerate(inplane_periodic):
            if is_periodic:
                continue
            selection_basis[row_index, row_index + 1] = 1.0

        return selection_basis

    def __x_index_range(
        self,
        primitive_periods: np.ndarray,
        rotated_unit_cell_basis: np.ndarray,
        x_bounds: np.ndarray,
    ) -> np.ndarray:
        """
        Build a conservative contiguous lattice-index range along the x-period vector.

        The x-period direction is derived in lattice space as the cross product of the
        two in-plane primitive periods expressed in the rotated unit-cell basis. The
        returned integer range is padded conservatively so translated unit cells cover
        the requested x slab after in-plane box tilts and unit-cell extent are applied.

        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :param rotated_unit_cell_basis: 3x3 array containing the rotated unit-cell
            basis vectors as rows.
        :param x_bounds: Length-2 array-like containing ``[x_min, x_max]``.
        :return: Contiguous integer array of lattice indices along the x-period
            direction.
        """
        primitive_periods = np.asarray(primitive_periods, dtype=np.float64)
        rotated_unit_cell_basis = np.asarray(
            rotated_unit_cell_basis, dtype=np.float64
        )
        x_bounds = np.asarray(x_bounds, dtype=np.float64)

        determinant = np.linalg.det(rotated_unit_cell_basis)
        if np.isclose(determinant, 0.0, atol=self.__epsilon, rtol=0.0):
            raise GBMakerValueError(
                "rotated_unit_cell_basis must form an invertible 3x3 basis."
            )

        reduced_periods = np.linalg.solve(
            rotated_unit_cell_basis.T, primitive_periods.T
        ).T
        x_direction_lattice = np.cross(reduced_periods[0], reduced_periods[1])
        if np.linalg.norm(x_direction_lattice) <= self.__epsilon:
            raise GBMakerValueError(
                "primitive_periods must define distinct in-plane directions."
            )

        rounded_direction = np.rint(x_direction_lattice)
        if np.allclose(
            x_direction_lattice, rounded_direction, atol=self.__epsilon, rtol=0.0
        ) and np.any(rounded_direction):
            x_direction_lattice = self.__reduce_integer_row(
                rounded_direction.astype(int)
            ).astype(np.float64)

        x_period_vector = x_direction_lattice @ rotated_unit_cell_basis
        x_projection = float(x_period_vector[0])
        if np.isclose(x_projection, 0.0, atol=self.__epsilon, rtol=0.0):
            raise GBMakerValueError(
                "x-period direction must have a non-zero projection on x."
            )
        if x_projection < 0.0:
            x_projection = -x_projection

        box_basis = self.__box_periodic_basis(primitive_periods)
        box_corners_x = np.array(
            [
                0.0,
                box_basis[0, 0],
                box_basis[1, 0],
                box_basis[0, 0] + box_basis[1, 0],
            ],
            dtype=np.float64,
        )
        cell_corners_x = np.array(
            [
                np.sum(
                    rotated_unit_cell_basis[np.array(mask, dtype=bool), 0],
                    dtype=np.float64,
                )
                for mask in np.ndindex((2, 2, 2))
            ],
            dtype=np.float64,
        )

        x_offset_min = float(np.min(box_corners_x) + np.min(cell_corners_x))
        x_offset_max = float(np.max(box_corners_x) + np.max(cell_corners_x))

        n_min = math.floor((x_bounds[0] - x_offset_max) / x_projection) - 1
        n_max = math.ceil((x_bounds[1] - x_offset_min) / x_projection) + 1
        return np.arange(n_min, n_max + 1, dtype=int)

    def __reduced_box_coordinates(
        self, cartesian_coordinates: np.ndarray, box_basis: np.ndarray
    ) -> np.ndarray:
        """
        Convert Cartesian coordinates to mixed box coordinates ``[x_cart, u_y, u_z]``.

        The mixed basis is ``[e_x, A_y, A_z]`` where ``e_x`` is the Cartesian x-axis
        and ``A_y``/``A_z`` are the in-plane box basis vectors.

        :param cartesian_coordinates: Cartesian coordinates with shape ``(..., 3)``.
        :param box_basis: 2x3 array containing ``A_y`` and ``A_z``.
        :return: Mixed box coordinates with shape ``(..., 3)``.
        """
        cartesian_coordinates = np.asarray(cartesian_coordinates, dtype=np.float64)
        box_basis = np.asarray(box_basis, dtype=np.float64)

        yz_basis = box_basis[:, 1:].T
        determinant = np.linalg.det(yz_basis)
        if np.isclose(determinant, 0.0, atol=self.__epsilon, rtol=0.0):
            raise GBMakerValueError(
                "box_basis y/z projections must form an invertible 2x2 basis."
            )

        yz_coordinates = cartesian_coordinates[..., 1:]
        reduced_yz = np.linalg.solve(
            yz_basis, yz_coordinates.reshape(-1, 2).T
        ).T.reshape(yz_coordinates.shape)
        x_cart = (
            cartesian_coordinates[..., 0]
            - reduced_yz[..., 0] * box_basis[0, 0]
            - reduced_yz[..., 1] * box_basis[1, 0]
        )
        return np.concatenate((x_cart[..., np.newaxis], reduced_yz), axis=-1)

    def __cartesian_from_box_coordinates(
        self, box_coordinates: np.ndarray, box_basis: np.ndarray
    ) -> np.ndarray:
        """
        Convert mixed box coordinates ``[x_cart, u_y, u_z]`` to Cartesian coordinates.

        :param box_coordinates: Mixed box coordinates with shape ``(..., 3)``.
        :param box_basis: 2x3 array containing ``A_y`` and ``A_z``.
        :return: Cartesian coordinates with shape ``(..., 3)``.
        """
        box_coordinates = np.asarray(box_coordinates, dtype=np.float64)
        box_basis = np.asarray(box_basis, dtype=np.float64)

        cartesian_coordinates = np.array(box_coordinates, copy=True)
        cartesian_coordinates[..., 0] += np.tensordot(
            box_coordinates[..., 1:], box_basis[:, 0], axes=([-1], [0])
        )
        cartesian_coordinates[..., 1:] = np.tensordot(
            box_coordinates[..., 1:], box_basis[:, 1:], axes=([-1], [0])
        )
        return cartesian_coordinates

    def __clip_atoms_to_cartesian_box(
        self, atoms: np.ndarray, x_bounds: np.ndarray
    ) -> np.ndarray:
        """
        Clip atoms to Cartesian box bounds on non-periodic axes.

        X is always clipped to the slab bounds. Y and Z are only clipped when the
        corresponding in-plane axis is non-periodic. Small negative y/z values within
        epsilon are snapped to zero after filtering.

        :param atoms: Structured atom array containing ``x``, ``y``, and ``z`` fields.
        :param x_bounds: Length-2 array-like containing ``[x_min, x_max]``.
        :return: Filtered atom array, with lower-face y/z values clamped to zero on
            non-periodic axes.
        """

        x_bounds = np.asarray(x_bounds, dtype=np.float64)

        inplane_periodic = self.__inplane_periodic
        inside_box = (atoms["x"] >= x_bounds[0] -
                      self.__epsilon) & (atoms["x"] < x_bounds[1] - self.__epsilon)

        axis_names = ("y", "z")
        axis_dims = (self.__y_dim, self.__z_dim)
        for axis_name, axis_dim, is_periodic in zip(axis_names, axis_dims, inplane_periodic):
            if is_periodic:
                continue
            inside_box &= (
                (atoms[axis_name] >= -self.__epsilon) & (atoms[axis_name] < axis_dim)
            )

        clipped_atoms = atoms[inside_box].copy()
        for axis_name, is_periodic in zip(axis_names, inplane_periodic):
            if is_periodic:
                continue
            clipped_atoms[axis_name] = np.where(
                (clipped_atoms[axis_name] < 0.0) & (
                    clipped_atoms[axis_name] >= -self.__epsilon),
                0.0,
                clipped_atoms[axis_name],
            )

        return clipped_atoms

    def __deduplicate_positions(self, atoms: np.ndarray) -> np.ndarray:
        """
        Remove duplicate atoms using epsilon-quantized Cartesian positions.
        Keeps the first occurrence of each position.
        """
        if len(atoms) == 0:
            return atoms

        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        quantized = np.round(positions / self.__epsilon).astype(np.int64)
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        return atoms[np.sort(unique_indices)]

    def __select_atoms_in_box_basis(
        self,
        atoms: np.ndarray,
        primitive_periods: np.ndarray,
        x_bounds: np.ndarray,
    ) -> np.ndarray:
        """
        Select atoms using mixed box coordinates for in-plane boundary handling.

        Periodic in-plane axes are filtered in reduced coordinates on the half-open
        interval ``[0, 1)`` up to tolerance, wrapped back into the canonical cell, then
        mapped back to Cartesian coordinates. Non-periodic axes remain Cartesian in the
        mixed basis and are clipped against the box dimensions directly.

        :param atoms: Structured atom array containing ``x``, ``y``, and ``z`` fields.
        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :param x_bounds: Length-2 array-like containing ``[x_min, x_max]``.
        :return: Filtered structured atom array with wrapped in-plane coordinates.
        """
        x_bounds = np.asarray(x_bounds, dtype=np.float64)

        selection_basis = self.__selection_basis_vectors(primitive_periods)
        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        inplane_periodic = self.__inplane_periodic

        # Fast path: orthogonal box - no tilt, so in-plane basis vectors have zero
        # x-projection. Reduced coordinates are simple ratios; skip the solve round trip.
        if np.allclose(selection_basis[:, 0], 0.0, atol=self.__epsilon, rtol=0.0):
            inside_box = np.ones(len(atoms), dtype=bool)
            for row_index, is_periodic in enumerate(inplane_periodic):
                axis = row_index + 1  # 1=y, 2=z
                period = selection_basis[row_index, axis]
                coord = positions[:, axis]
                if is_periodic:
                    tol = self.__reduced_coordinate_tolerance(
                        selection_basis[row_index])
                    inside_box &= (coord / period >= -
                                   tol) & (coord / period < 1.0 + tol)
                # non-periodic handled by __clip_atoms_to_cartesian_box
            selected_atoms = atoms[inside_box].copy()
            if len(selected_atoms) == 0:
                return selected_atoms

            for row_index, is_periodic in enumerate(inplane_periodic):
                if not is_periodic:
                    continue
                axis = row_index + 1
                period = selection_basis[row_index, axis]
                tol = self.__reduced_coordinate_tolerance(selection_basis[row_index])
                selected_atoms_coord = selected_atoms[("y", "z")[row_index]]
                wrapped = np.mod(selected_atoms_coord, period)
                selected_atoms[("y", "z")[row_index]] = np.where(
                    (wrapped < tol * period) | ((period - wrapped) < tol * period),
                    0.0,
                    wrapped,
                )
            inside_x = (
                (selected_atoms["x"] >= x_bounds[0] - self.__epsilon)
                & (selected_atoms["x"] < x_bounds[1] - self.__epsilon)
            )
            selected_atoms = selected_atoms[inside_x]
            if len(selected_atoms) == 0:
                return selected_atoms
            return self.__deduplicate_positions(selected_atoms)

        box_coordinates = self.__reduced_box_coordinates(positions, selection_basis)

        inside_box = np.ones(len(atoms), dtype=bool)
        axis_dims = (self.__y_dim, self.__z_dim)
        for row_index, (axis_dim, is_periodic) in enumerate(
            zip(axis_dims, inplane_periodic)
        ):
            reduced_axis = box_coordinates[:, row_index + 1]
            if is_periodic:
                tol = self.__reduced_coordinate_tolerance(selection_basis[row_index])
                inside_box &= (
                    (reduced_axis >= -tol) & (reduced_axis < 1.0 + tol)
                )
            else:
                inside_box &= (
                    (reduced_axis >= -self.__epsilon) & (reduced_axis < axis_dim)
                )

        selected_atoms = atoms[inside_box].copy()
        if len(selected_atoms) == 0:
            return selected_atoms

        selected_box_coordinates = box_coordinates[inside_box].copy()
        for row_index, is_periodic in enumerate(inplane_periodic):
            coordinate_index = row_index + 1
            if is_periodic:
                tol = self.__reduced_coordinate_tolerance(selection_basis[row_index])
                selected_box_coordinates[:, coordinate_index] = wrap_reduced_coordinate(
                    selected_box_coordinates[:, coordinate_index],
                    tol,
                )
                continue

            selected_box_coordinates[:, coordinate_index] = np.where(
                (
                    (selected_box_coordinates[:, coordinate_index] < 0.0)
                    & (selected_box_coordinates[:, coordinate_index] >= -self.__epsilon)
                ),
                0.0,
                selected_box_coordinates[:, coordinate_index],
            )

        wrapped_positions = self.__cartesian_from_box_coordinates(
            selected_box_coordinates, selection_basis
        )
        selected_atoms["x"], selected_atoms["y"], selected_atoms["z"] = wrapped_positions.T

        inside_x = (
            (selected_atoms["x"] >= x_bounds[0] - self.__epsilon)
            & (selected_atoms["x"] < x_bounds[1] - self.__epsilon)
        )
        selected_atoms = selected_atoms[inside_x]
        if len(selected_atoms) == 0:
            return selected_atoms

        return self.__deduplicate_positions(selected_atoms)

    def __apply_strain_accommodation(
        self,
        current_dim: float,
        axis_name: str,
        row_left,
        row_right,
    ) -> float:
        """Return a commensurate in-plane dimension for one axis.

        Calls ``_find_commensurate_pair`` with ``self.__mismatch_tol`` and
        ``self.__mismatch_max_cells``.  If a pair is found, the returned
        dimension is chosen according to ``self.__strain_grain``:

        - ``"both"`` — average of n1*d1 and n2*d2 (symmetric strain on each)
        - ``"left"`` — n2*d2 (right grain exact, left grain absorbs strain)
        - ``"right"`` — n1*d1 (left grain exact, right grain absorbs strain)

        If no pair is found, emits a ``UserWarning`` and returns
        ``current_dim`` unchanged.

        :param current_dim: Current box dimension (Angstroms).
        :param axis_name: ``"y"`` or ``"z"`` (used in the warning message).
        :param row_left: Row of ``__R_left_approx`` for this axis.
        :param row_right: Row of ``__R_right_approx`` for this axis.
        :return: Possibly adjusted box dimension.
        """
        d1 = self.__a0 * float(np.linalg.norm(np.asarray(row_left, dtype=float)))
        d2 = self.__a0 * float(np.linalg.norm(np.asarray(row_right, dtype=float)))
        result = _find_commensurate_pair(
            d1, d2, self.__mismatch_tol, self.__mismatch_max_cells
        )
        if result is None:
            residual = abs(d1 - d2) / max(d1, d2)
            warnings.warn(
                f"No commensurate {axis_name} pair found within "
                f"mismatch_max_cells={self.__mismatch_max_cells} for "
                f"mismatch_tol={self.__mismatch_tol}. "
                f"Residual mismatch: {residual:.4%}. "
                f"Falling back to max(d1, d2) * repeat_factor.",
                UserWarning,
                stacklevel=4,
            )
            return current_dim
        n1, n2, l1, l2 = result
        if self.__strain_grain == "both":
            return (l1 + l2) / 2.0
        if self.__strain_grain == "left":
            return l2
        return l1  # "right"

    def __update_dims(self) -> None:
        """
        Updates the y_dim and z_dim parameters after a relevant parameter has been
        changed.

        :raises UserWarning: Warning issued when the repeat factors are modified to
            accommodate the interaction distance.
        """
        self.__y_dim = self.__repeat_factor[0] * self.__spacing["y"]
        self.__z_dim = self.__repeat_factor[1] * self.__spacing["z"]

        use_exact = (
            self.__embedding is not None
            and self.__embedding.exact
            and self.__embedding.P is not None
        )
        if self.__mismatch_tol is not None and not use_exact:
            self.__y_dim = self.__apply_strain_accommodation(
                self.__y_dim, "y",
                self.__R_left_approx[1], self.__R_right_approx[1],
            )
            self.__z_dim = self.__apply_strain_accommodation(
                self.__z_dim, "z",
                self.__R_left_approx[2], self.__R_right_approx[2],
            )

        repeat_z = self.__repeat_factor[1]
        cutoff = 2 * self.__interaction_distance
        if self.__y_dim < cutoff:
            repeat_y = math.ceil(cutoff / self.__spacing["y"])
            self.__y_dim = repeat_y * self.__spacing["y"]
            warnings.warn(
                f"Repeat factor in y modified to {repeat_y} to accommodate interaction "
                f"distance of {cutoff}."
            )
            self.__repeat_factor[0] = repeat_y
        if self.__z_dim < cutoff:
            repeat_z = math.ceil(cutoff / self.__spacing["z"])
            self.__z_dim = repeat_z * self.__spacing["z"]
            warnings.warn(
                f"Repeat factor in z modified to {repeat_z} to accommodate interaction "
                f"distance of {cutoff}."
            )
            self.__repeat_factor[1] = repeat_z
        self.__box_dims = self.__calculate_box_dimensions()

        self.__generate_gb()
        self.__set_gb_region()

    def __validate(
        self,
        value: Any,
        expected_types: Union[type, Tuple[type, ...]],
        parameter_name: str,
        *,
        positive: bool = False,
        expected_length: int | None = None,
        strictly_positive: bool = False
    ):
        """
        Private method for validating the values passed in using the setters.

        :param value: The value to validate.
        :param expected_types: Single type or tuple containing the valid types for
            value.
        :param parameter_name: The name of the parameter.
        :param positive: Whether or not the value should be positive (>= 0), optional,
            defaults to False.
        :param expected_length: Specific to sequences or arrays. The expected length of
            the sequence or array, optional, defaults to None.
        :param strictly_positive: Supercedes `positive` by enforcing value > 0. Optional,
            defaults to False.
        :raises GBMakerTypeError: Exception raised if the type of the value does not
            match the expected type(s).
        :raises GBMakerValueError: Exception raised when invalid values are given for
            the specified parameter.
        :return: The validated value.
        """
        if not isinstance(expected_types, tuple):
            expected_types = (expected_types,)
        if not any(isinstance(value, t) for t in expected_types) and not isinstance(
            value, np.generic
        ):
            expected_type_names = ", ".join(t.__name__ for t in expected_types)
            raise GBMakerTypeError(
                f"{parameter_name} must be of type {expected_type_names}."
            )

        if strictly_positive and isinstance(value, Number):
            if value <= 0:
                raise GBMakerValueError(f"{parameter_name} must be strictly positive")
            if value < np.finfo(np.float64).eps:
                warnings.warn(
                    f"{parameter_name} ({value}) is below machine epsilon "
                    f"({np.finfo(np.float64).eps:.2e}) and may not have any "
                    "practical effect."
                )
        elif positive and isinstance(value, Number) and value < 0:
            raise GBMakerValueError(
                f"{parameter_name} must be a positive value.")

        if (
            isinstance(value, (Sequence, np.ndarray))
            and all([isinstance(val, Number) for val in value])
            and positive
        ):
            for val in value:
                if val < 0:
                    raise GBMakerValueError(
                        f"{parameter_name} must have all positive values."
                    )

        if (
            expected_length is not None
            and isinstance(value, (Sequence, np.ndarray))
            and len(value) != expected_length
        ):
            raise GBMakerValueError(
                f"{parameter_name} must have {expected_length} elements."
            )

        if parameter_name == "structure" and value not in [
            "fcc",
            "bcc",
            "sc",
            "diamond",
            "fluorite",
            "rocksalt",
            "zincblende",
        ]:
            raise GBMakerValueError(
                f"{parameter_name} ({value}) must be one of ['fcc', 'bcc', 'sc', "
                + "'diamond', 'fluorite', 'rocksalt', 'zincblende']."
            )

        if parameter_name == "repeat_factor":
            if isinstance(value, int):
                if value < 2:
                    warnings.warn("Recommended repeat distance at least 2.")
                value = [value, value]
            else:  # isinstance(value, (Sequence, np.ndarray))
                for val in value:
                    if not isinstance(val, int):
                        raise GBMakerValueError(
                            "repeat_factor must be a sequence of type int."
                        )
                    if val < 2:
                        warnings.warn(
                            "Recommended repeat distance is at least 2.")
                value = list(value)
        return value

    # Public methods
    def get_supercell(self, corners: np.ndarray) -> np.ndarray:
        """
        Generates a supercell of lattice sites.

        :param corners: Array containing the position of the corners of the unit cells.
        :return: Structured numpy array containing the atom data (type and position) for
            the supercell.
        """
        # Unit cell as structured array
        unit_cell = self.__unit_cell.asarray()
        supercell = np.tile(unit_cell, len(corners))
        translations = np.repeat(corners, len(unit_cell), axis=0)
        supercell["x"] += translations[:, 0]
        supercell["y"] += translations[:, 1]
        supercell["z"] += translations[:, 2]
        return supercell

    def update_spacing(self, threshold: float = None) -> None:
        """
        Update the periodic spacing based on the rotation matrix and the optional
        threshold parameter.

        :param threshold: The maximum allowed value that any spacing can take
        """
        self.__spacing = self.__calculate_periodic_spacing(threshold)
        self.__update_dims()

    def write_lammps(
        self,
        file_name: str,
        atoms: np.ndarray = None,
        box_sizes: np.ndarray = None,
        *,
        type_as_int: bool = False,
        precision: int = 6,
        charges: dict = None,
        triclinic: bool = False
    ) -> None:
        """
        Writes the atom positions with the given box dimensions to a LAMMPS input file.

        :param str file_name: The filename to save the data
        :param np.ndarray atoms: The numpy array containing the atom data.
        :param np.ndarray box_sizes: 3x2 array containing the min and max dimensions for
            each of the x, y, and z dimensions.
        :param type_as_int: Whether to write the atom types as a chemical name or a
            number. Keyword argument, optional, defaults to False (write as a chemical
            name).
        :param precision: The decimal precision to use when writing float values,
            optional, default = 6.
        :param charges: dict containing the charge values for each type. Keys are
            expected to be integers, values are expected to be numeric. Optional,
            default is None.
        """
        if not isinstance(file_name, str):
            raise GBMakerTypeError("file_name must be of type str")
        if atoms is None and box_sizes is None:
            atoms = self.__whole_system
            box_sizes = self.__box_dims
        elif (atoms is None and box_sizes is not None) or (
            atoms is not None and box_sizes is None
        ):
            raise GBMakerValueError(
                "'atoms' and 'box_sizes' must be specified together."
            )

        atom_names = np.unique(atoms["name"])
        if set(atom_names).issubset(self.__unit_cell.type_map.keys()):
            name_to_int = {
                name: self.__unit_cell.type_map[name]
                for name in self.__unit_cell.type_map
                if name in atom_names
            }
        else:
            name_to_int = {name: i + 1 for i, name in enumerate(atom_names)}

        if charges is not None:
            if not all([isinstance(i, int) or isinstance(i, str) for i in charges.keys()]):
                raise GBMakerValueError(
                    "'charges' keys are required to be integers or strings.")
            if not all([isinstance(i, Number) for i in charges.values()]):
                raise GBMakerValueError(
                    "'charges' values are required to be numeric.")
            if type_as_int:
                if all([isinstance(i, str) for i in charges.keys()]):
                    for name in np.unique(atoms["name"]):
                        charges[name_to_int[name]] = charges[name]

        def format_atom_line(index, name, pos, charge=None):
            if type_as_int:
                name = name_to_int[name]
            if charge is not None:
                return (f"{index} {name} {charge:.{precision}f} " +
                        f"{pos[0]:.{precision}f} {pos[1]:.{precision}f} " +
                        f"{pos[2]:.{precision}f}\n")
            else:
                return (f"{index} {name} {pos[0]:.{precision}f} " +
                        f"{pos[1]:.{precision}f} {pos[2]:.{precision}f}\n")

        # Write LAMMPS data file
        with open(file_name, "w") as fdata:
            # First line is a comment line
            atom_names = "".join(np.unique(atoms["name"]))
            fdata.write(f"Crystalline {atom_names} atoms\n\n")

            # --- Header ---#
            # Specify number of atoms and atom types
            fdata.write("{} atoms\n".format(len(atoms)))
            fdata.write("{} atom types\n".format(len(set(atoms["name"]))))
            # Specify box dimensions
            fdata.write(
                f'{box_sizes[0][0]:.{precision}f} {box_sizes[0][1]:.{precision}f} xlo xhi\n')
            fdata.write(
                f'{box_sizes[1][0]:.{precision}f} {box_sizes[1][1]:.{precision}f} ylo yhi\n')
            fdata.write(
                f'{box_sizes[2][0]:.{precision}f} {box_sizes[2][1]:.{precision}f} zlo zhi\n')
            if triclinic:
                xy, xz, yz, theta = self.__get_triclinic_params()
                fdata.write(
                    f'{xy:.{precision}f} {xz:.{precision}f} {yz:.{precision}f} xy xz yz\n'
                )
                ct, st = math.cos(theta), math.sin(theta)
                Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])

            if not type_as_int:
                fdata.write("\nAtom Type Labels\n\n")
                for name, value in name_to_int.items():
                    fdata.write(f"{value} {name}\n")

            # Atoms section
            fdata.write("\nAtoms\n\n")

            # Write each position.
            for i, (name, *pos) in enumerate(atoms):
                if charges is not None:
                    charge = charges[name_to_int[name]
                                     ]if type_as_int else charges[name]
                else:
                    charge = None

                if triclinic:
                    pos = Rx @ np.array(pos, dtype=float)
                fdata.write(format_atom_line(i + 1, name, pos, charge))

    # Properties with getters and setters. Automatic updates for related parameters are
    # automatically taken care of.
    @property
    def a0(self) -> float:
        return self.__a0

    @a0.setter
    def a0(self, value: Number) -> None:
        atom_types = tuple(self.__unit_cell.names())
        self.__a0 = self.__validate(value, float, "a0", positive=True)
        self.__unit_cell = self.__init_unit_cell(atom_types)
        self.update_spacing()

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value: Number) -> None:
        self.__epsilon = self.__validate(
            value, Number, "epsilon", strictly_positive=True)

    @property
    def gb_thickness(self) -> float:
        return self.__gb_thickness

    @gb_thickness.setter
    def gb_thickness(self, value: Number):
        self.__gb_thickness = self.__validate(
            value, Number, "gb_thickness", positive=True)
        self.__box_dims = self.__calculate_box_dimensions()

    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, value: int):
        self.__id = self.__validate(value, int, "id", positive=True)

    @property
    def interaction_distance(self) -> float:
        return self.__interaction_distance

    @interaction_distance.setter
    def interaction_distance(self, value: Number) -> None:
        self.__interaction_distance = self.__validate(
            value, Number, "interaction_distance", positive=True)
        self.__update_dims()

    @property
    def misorientation(self) -> np.ndarray:
        return np.hstack((self.__misorientation, self.__inclination))

    @misorientation.setter
    def misorientation(self, value: np.ndarray):
        misorientation = self.__validate(
            value, np.ndarray, "misorientation", expected_length=5
        )
        self.__misorientation = misorientation[:3]
        self.__inclination = misorientation[3:]
        self.__Rmis = Rotation.from_euler(
            "ZXZ", misorientation[:3]).as_matrix()
        self.__Rincl = (
            Rotation.from_euler("z", misorientation[4])
            * Rotation.from_euler("y", misorientation[3])
        ).as_matrix()
        # Discard any active embedding so update_spacing uses the new Euler
        # angles rather than the stale embedding-derived rotation matrices.
        self.__embedding = None
        self.update_spacing()

    @property
    def repeat_factor(self) -> int:
        return self.__repeat_factor

    @repeat_factor.setter
    def repeat_factor(self, value: int):
        self.__repeat_factor = self.__validate(
            value, (int, Sequence), "repeat_factor", positive=True)
        self.__update_dims()

    @property
    def structure(self) -> str:
        return self.__structure

    @structure.setter
    def structure(self, value: str) -> None:
        self.__structure = self.__validate(value, str, "structure")
        if set([self.__structure, value]).issubset(
            set(["fluorite", "rocksalt", "zincblende"])
        ):
            raise GBMakerValueError(
                f"Cannot estimate conversion from {self.__structure} to {value}"
            )
        else:
            atom_types = tuple(set(self.__unit_cell.names()))

        self.__unit_cell = self.__init_unit_cell(atom_types)

    @property
    def vacuum_thickness(self) -> int:
        return self.__vacuum_thickness

    @vacuum_thickness.setter
    def vacuum_thickness(self, value: Number):
        old_vacuum = self.__vacuum_thickness
        self.__vacuum_thickness = self.__validate(
            value, Number, "vacuum_thickness", positive=True
        )
        delta = self.__vacuum_thickness - old_vacuum
        self.__left_grain["x"] += delta
        self.__right_grain["x"] += delta
        self.__whole_system["x"] += delta
        self.__gb_region["x"] += delta
        self.__box_dims = self.__calculate_box_dimensions()

    @property
    def x_dim_min(self) -> np.ndarray:
        return self.__x_dim_min

    @x_dim_min.setter
    def x_dim_min(self, value: Number):
        self.__x_dim_min = self.__validate(
            value, Number, "x_dim_min", positive=True)
        self.update_spacing()
        self.__box_dims = self.__calculate_box_dimensions()

    # Additional getters for other class properties
    @property
    def inplane_periodic(self) -> tuple:
        """Read-only view of the in-plane periodicity flags (y, z)."""
        return tuple(bool(v) for v in self.__inplane_periodic)

    @property
    def box_dims(self) -> np.ndarray:
        return self.__box_dims

    @property
    def whole_system(self) -> np.ndarray:
        return self.__whole_system

    @property
    def left_grain(self) -> np.ndarray:
        return self.__left_grain

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def right_grain(self) -> np.ndarray:
        return self.__right_grain

    @property
    def gb_plane_x(self) -> float:
        return self.__vacuum_thickness + self.__left_x

    @property
    def spacing(self) -> dict:
        return self.__spacing

    @property
    def unit_cell(self) -> UnitCell:
        return self.__unit_cell

    @property
    def x_dim(self) -> float:
        return self.__x_dim

    @property
    def y_dim(self) -> float:
        return self.__y_dim

    @property
    def z_dim(self) -> float:
        return self.__z_dim
