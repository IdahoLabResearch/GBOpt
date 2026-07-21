# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Public exact boundary and CSL utilities.

Import user-facing exact utilities from this package directly, for example::

    from GBOpt.crystallography import csl_exact_spec_to_embedding, ``canonicalize_pq``
"""

from .boundary import (
    csl_approx_spec_to_embedding,
    csl_exact_spec_to_embedding,
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
    validate_orientation_matrix,
)
from .pq import (
    canonicalize_pq,
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from .quaternion import (
    normalize_integer_quaternion,
    quaternion_to_scaled_rotation,
)
from .types import (
    CoincidenceCheck,
    CSLResult,
    DSCBasis,
    InPlaneBasis,
    ScaledRotation,
    SmithDiagnostics,
)

__all__ = [
    "csl_approx_spec_to_embedding",
    "csl_exact_spec_to_embedding",
    "pq_spec_to_embedding",
    "primitive_bicrystal_atom_count",
    "csl_from_scaled_rotation",
    "dsc_basis",
    "exactify_five_dof",
    "build_mixed_orientations",
    "build_symmetric_tilt_orientations",
    "build_tilt_orientations",
    "build_twist_orientations",
    "five_dof_from_axis_angle",
    "five_dof_from_orientation_matrices",
    "inclination_from_normal",
    "normalize_direction",
    "validate_orientation_matrix",
    "canonicalize_pq",
    "canonicalize_pq_paired",
    "recover_exact_row_rotation_from_paired_pq",
    "CoincidenceCheck",
    "CSLResult",
    "DSCBasis",
    "InPlaneBasis",
    "ScaledRotation",
    "SmithDiagnostics",
    "normalize_integer_quaternion",
    "quaternion_to_scaled_rotation",
]
