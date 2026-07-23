# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Public crystallography types, boundary adapters, and construction utilities."""

from .boundary import (
    csl_approx_spec_to_embedding,
    csl_exact_spec_to_embedding,
    five_dof_spec_to_embedding,
    pq_spec_to_embedding,
    primitive_bicrystal_atom_count,
)
from .csl import (
    csl_from_scaled_rotation,
    dsc_basis,
)
from .exactification import exactify_five_dof
from .orientation import (
    build_mixed_orientations,
    build_symmetric_tilt_orientations,
    build_tilt_orientations,
    build_twist_orientations,
    five_dof_from_axis_angle,
    five_dof_from_orientation_matrices,
    inclination_from_normal,
    normalize_direction,
    orientation_matrices_from_five_dof,
    validate_orientation_matrix,
)
from .pq import (
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from .quaternion import (
    normalize_integer_quaternion,
    quaternion_to_scaled_rotation,
)
from .types import (
    CoincidenceCheck,
    CrystallographyBackendError,
    CrystallographyDivisibilityError,
    CrystallographyError,
    CrystallographyNotImplementedError,
    CrystallographyValueError,
    CSLResult,
    DSCBasis,
    InPlaneBasis,
    ScaledRotation,
    SmithDiagnostics,
)

__all__ = [
    # Exceptions
    "CrystallographyError",
    "CrystallographyValueError",
    "CrystallographyBackendError",
    "CrystallographyDivisibilityError",
    "CrystallographyNotImplementedError",

    # Result and domain types
    "ScaledRotation",
    "SmithDiagnostics",
    "CSLResult",
    "InPlaneBasis",
    "DSCBasis",
    "CoincidenceCheck",

    # Boundary-spec adapters
    "pq_spec_to_embedding",
    "csl_exact_spec_to_embedding",
    "csl_approx_spec_to_embedding",
    "five_dof_spec_to_embedding",
    "primitive_bicrystal_atom_count",

    # CSL and DSC construction
    "csl_from_scaled_rotation",
    "dsc_basis",

    # Exactification
    "exactify_five_dof",

    # Orientation construction and conversion
    "normalize_direction",
    "validate_orientation_matrix",
    "build_tilt_orientations",
    "build_symmetric_tilt_orientations",
    "build_twist_orientations",
    "build_mixed_orientations",
    "inclination_from_normal",
    "five_dof_from_axis_angle",
    "five_dof_from_orientation_matrices",
    "orientation_matrices_from_five_dof",

    # P/Q operations
    "canonicalize_pq_paired",
    "recover_exact_row_rotation_from_paired_pq",

    # Quaternion operations
    "normalize_integer_quaternion",
    "quaternion_to_scaled_rotation",
]
