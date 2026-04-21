# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Grain boundary builder utilities."""

import math
import warnings
from numbers import Number
from typing import Any, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from GBOpt.UnitCell import UnitCell


class GBMakerError(Exception):
    """Base class for Exceptions in the GBMaker class."""
    pass


class GBMakerTypeError(GBMakerError):
    """Exception raised when an invalid type is assigned to a GBMaker attribute."""
    pass


class GBMakerValueError(GBMakerError):
    """Exception raised when an invalid value is assigned to a GBMaker attribute."""
    pass


def wrap_reduced_coordinate(reduced_coord: np.ndarray, tol: float) -> np.ndarray:
    """
    Wrap reduced coordinates into [0, 1) and snap both periodic faces to 0.

    :param reduced_coord: Reduced coordinates to wrap.
    :param tol: Tolerance in reduced-coordinate units.
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

        self.__unit_cell = self.__init_unit_cell(atom_types)
        self.__spacing = self.__calculate_periodic_spacing()  # periodic distances dict
        self.__update_dims()

        self.__radius = a0 * self.__unit_cell.radius  # atom radius
        self.__generate_gb()
        self.__set_gb_region()
        self.__box_dims = self.__calculate_box_dimensions()

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

    def __orient_period_rows(self, R_grain: np.ndarray, approx: np.ndarray) -> np.ndarray:
        """
        Orient approximate periodic rows to match the corresponding grain rows.

        The integer approximation preserves row norms, but the sign of each row is
        arbitrary. Flip any row whose direction is antiparallel to the matching float
        row so the periodic basis remains consistently oriented.

        :param R_grain: The grain rotation matrix in floating-point form.
        :param approx: Integer approximation of the same matrix.
        :return: A copy of ``approx`` with row signs oriented to ``R_grain``.
        """
        R_grain = np.asarray(R_grain, dtype=np.float64)
        oriented = np.array(approx, copy=True)
        for i, grain_row in enumerate(R_grain):
            approx_row = np.asarray(oriented[i], dtype=np.float64)
            if np.dot(grain_row, approx_row) < 0:
                oriented[i] = -oriented[i]
        return oriented

    # Private class methods
    def __approximate_rotation_row_as_int(
        self, row: np.ndarray, angle_tol_deg: float = 0.5, max_scale: int = 10000
    ) -> np.ndarray:
        """
        Find the smallest-norm integer vector that points within *angle_tol_deg* of
        *row*. LLL lattice reduction was considered but not adopted: for CSL boundaries
        the correct scale factor k equals ‖g‖ ≤ √Σ (typically < 10), so the loop exits
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

    def __generate_gb(self) -> None:
        """
        Private method to calculate the left grain, right grain, and whole GB system
        """
        self.__left_grain = self.__generate_left_grain()
        self.__right_grain = self.__generate_right_grain()
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

        # approximate the rotation matrix as integers
        self.__R_left = self.__Rincl
        self.__R_right = np.dot(self.__Rincl, self.__Rmis)
        # # We store the approximate matrices as objects to allow for large numbers
        self.__R_left_approx = self.__approximate_rotation_matrix_as_int(
            self.__R_left).astype(object)
        self.__R_right_approx = self.__approximate_rotation_matrix_as_int(
            self.__R_right).astype(object)
        self.__R_left_approx = self.__orient_period_rows(
            self.__R_left, self.__R_left_approx).astype(object)
        self.__R_right_approx = self.__orient_period_rows(
            self.__R_right, self.__R_right_approx).astype(object)

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
        self.__x_dim = self.__left_x + self.__right_x
        spacing.update(
            {
                axis: max(spacing_left[axis], spacing_right[axis])
                for axis in ["y", "z"]
            }
        )

        warnings.simplefilter("once", UserWarning)
        for key, val in spacing.items():
            if key == 'x':
                continue
            if threshold < val:
                spacing[key] = threshold
                warnings.warn("Resulting boundary is non-periodic.")
        warnings.simplefilter("default", UserWarning)

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

        # Use grain with larger y-period, consistent with how spacing["y"] is chosen
        if np.linalg.norm(self.__R_left_approx[1]) >= np.linalg.norm(self.__R_right_approx[1]):
            R_grain = self.__R_left
            g_y = self.__R_left_approx[1].astype(float)
            g_z = self.__R_left_approx[2].astype(float)
        else:
            R_grain = self.__R_right
            g_y = self.__R_right_approx[1].astype(float)
            g_z = self.__R_right_approx[2].astype(float)

        # Lab-frame period vectors
        A2_lab = R_grain @ (g_y * self.__a0)
        A3_lab = R_grain @ (g_z * self.__a0)

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

    def __generate_left_grain(self) -> np.ndarray:
        """
        Generates the left grain of the GB system.

        :return: 4xn array containing the atom data (type and position) for the left
            grain.
        """
        body_diagonal = np.linalg.norm(
            [self.__left_x, self.__y_dim, self.__z_dim])
        body_diagonal -= body_diagonal % self.__a0
        X = np.arange(-body_diagonal, body_diagonal, self.__a0)

        corners = np.vstack(np.meshgrid(X, X, X)).reshape(3, -1).T
        atoms = self.get_supercell(corners)

        positions = np.vstack((atoms["x"], atoms["y"], atoms["z"])).T
        rotated_positions = np.dot(positions, self.__Rincl.T)
        atoms["x"], atoms["y"], atoms["z"] = rotated_positions.T
        atoms["x"] += self.__vacuum_thickness

        return self.__get_points_inside_box(
            atoms,
            [
                self.__vacuum_thickness, 0, 0,
                self.__left_x + self.__vacuum_thickness, self.__y_dim, self.__z_dim
            ]
        )

    def __generate_right_grain(self) -> np.ndarray:
        """
        Generates the right grain of the GB system.

        The right grain is registered to the nominal interface plane (``left_x +
        vacuum``), not to the rightmost surviving atom of the clipped left grain. The
        resulting interfacial phase is one valid DSC configuration; further
        exploration of admissible phases is left to the optimization workflow.

        :return: 4xn array containing the atom data (type and position)
            for the right grain.
        """
        body_diagonal = np.linalg.norm(
            [self.__right_x, self.__y_dim, self.__z_dim])
        body_diagonal -= body_diagonal % self.__a0
        X = np.arange(-body_diagonal, body_diagonal, self.__a0)

        corners = np.vstack(np.meshgrid(X, X, X)).reshape(3, -1).T
        atoms = self.get_supercell(corners)

        R = np.dot(self.__Rincl, self.__Rmis)
        positions = np.vstack((atoms["x"], atoms["y"], atoms["z"])).T
        rotated_positions = np.dot(positions, R.T)
        atoms["x"], atoms["y"], atoms["z"] = rotated_positions.T

        interface_x = self.__left_x + self.__vacuum_thickness
        atoms["x"] += interface_x
        return self.__get_points_inside_box(
            atoms,
            [
                interface_x, 0, 0,
                self.__x_dim + self.__vacuum_thickness, self.__y_dim, self.__z_dim
            ]
        )

    def __set_gb_region(self):
        """
        Identifies the atoms in the GB region based on the gb thickness.
        """
        left_x_max = max(self.__left_grain['x'])
        right_x_min = min(self.__right_grain['x'])
        left_cut = left_x_max - self.__gb_thickness / 2.0
        right_cut = right_x_min + self.__gb_thickness / 2.0
        left_gb = self.__left_grain[self.__left_grain['x'] > left_cut]
        right_gb = self.__right_grain[self.__right_grain['x'] < right_cut]
        self.__gb_region = np.hstack((left_gb, right_gb))

    def __get_points_inside_box(self, atoms: np.ndarray, box_dim: np.ndarray) -> np.ndarray:
        """
        Selects the lattice points that are inside the given box dimensions.

        :param atoms: Atoms to check. 4xm array containing types and positions.
        :param box_dim: Dimensions of box (x_min, y_min, z_min, x_max, y_max, z_max).
        :return: 4xn array containing the Atom positions inside the given box.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = box_dim

        # Epsilon is applied symmetrically to all three directions.
        # The inclusion window is [dim_min - eps, dim_max - eps) for each axis.
        #
        # Lower bounds (all axes): epsilon catches atoms that float just below a box
        #   face due to rotation-matrix floating-point noise.
        #
        # x upper bound: the same noise can place a left-grain terminal plane just
        # below x_max; without this inward shift it coincides with the first right-grain
        # plane registered at x_max by __generate_right_grain.
        # y/z upper bounds: standard strict-less-than for periodic-image exclusion
        inside_box = (
            (atoms["x"] >= x_min - self.__epsilon) &
            (atoms["x"] < x_max - self.__epsilon) &
            (atoms["y"] >= y_min - self.__epsilon) &
            (atoms["y"] < y_max) &
            (atoms["z"] >= z_min - self.__epsilon) &
            (atoms["z"] < z_max)
        )
        return atoms[inside_box]

    def __reduced_coordinate_tolerance(self, basis_vector: np.ndarray) -> float:
        """
        Convert the Cartesian epsilon to reduced-coordinate units for a basis vector.

        :param basis_vector: Cartesian basis vector used to define the coordinate scale.
        :return: Reduced-coordinate tolerance corresponding to ``self.__epsilon``.
        """
        basis_vector = np.asarray(basis_vector, dtype=np.float64)
        basis_length = np.linalg.norm(basis_vector)
        if basis_length <= 0.0 or not np.isfinite(basis_length):
            raise GBMakerValueError("basis_vector must have non-zero length.")
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
        raw_period_vector = np.asarray(period_vector)
        if np.issubdtype(raw_period_vector.dtype, np.bool_):
            raise GBMakerTypeError("period_vector must be a real length-3 vector.")
        if np.iscomplexobj(raw_period_vector):
            raise GBMakerTypeError("period_vector must be a real length-3 vector.")
        if raw_period_vector.shape != (3,):
            raise GBMakerValueError("period_vector must be a length-3 vector.")
        if isinstance(box_length, (bool, np.bool_)):
            raise GBMakerTypeError("box_length must be of type Number.")

        try:
            period_vector = np.asarray(period_vector, dtype=float)
        except TypeError as exc:
            raise GBMakerTypeError(
                "period_vector must be a real length-3 vector."
            ) from exc
        except ValueError as exc:
            raise GBMakerTypeError(
                "period_vector must be a real length-3 vector."
            ) from exc
        if not np.all(np.isfinite(period_vector)):
            raise GBMakerValueError("period_vector must contain only finite values.")

        if not isinstance(axis_index, (int, np.integer)) or isinstance(axis_index, bool):
            raise GBMakerValueError("axis_index must be 0, 1, or 2.")
        try:
            box_length = float(box_length)
        except TypeError as exc:
            raise GBMakerTypeError("box_length must be a real number.") from exc
        except ValueError as exc:
            raise GBMakerValueError("box_length must be of type Number.") from exc
        if not np.isfinite(box_length):
            raise GBMakerValueError("box_length must be finite.")
        if box_length <= 0.0:
            raise GBMakerValueError("box_length must be strictly positive.")
        axis_index = int(axis_index)
        if axis_index not in (0, 1, 2):
            raise GBMakerValueError("axis_index must be 0, 1, or 2.")

        axis_projection = period_vector[axis_index]
        if not np.isfinite(axis_projection) or axis_projection == 0.0:
            raise GBMakerValueError(
                "period_vector must have a non-zero projection on the selected axis."
            )

        scale = box_length / axis_projection
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
        if primitive_periods.shape != (2, 3):
            raise GBMakerValueError("primitive_periods must be a 2x3 array.")

        inplane_periodic = getattr(self, "_GBMaker__inplane_periodic", (True, True))
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

    def __update_dims(self) -> None:
        """
        Updates the y_dim and z_dim parameters after a relevant parameter has been
        changed.

        :raises UserWarning: Warning issued when the repeat factors are modified to
            accommodate the interaction distance.
        """
        self.__y_dim = self.__repeat_factor[0] * self.__spacing["y"]
        self.__z_dim = self.__repeat_factor[1] * self.__spacing["z"]
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
        self.__vacuum_thickness = self.__validate(
            value, Number, "vacuum_thickness", positive=True
        )
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


if __name__ == "__main__":
    theta = math.radians(36.869898)
    G = GBMaker(
        a0=3.61,
        structure="fcc",
        gb_thickness=10.0,
        misorientation=[theta, 0, 0, 0, -theta / 2],
        repeat_factor=[3, 9],
    )
    G.write_lammps(np.vstack((G.left_grain, G.right_grain)),
                   G.box_dims, "test1.dat")
