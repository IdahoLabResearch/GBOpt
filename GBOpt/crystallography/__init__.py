# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Public exact boundary and CSL utilities.

Import user-facing exact utilities from this package directly, for example::

    from GBOpt.crystallography import csl_spec_to_embedding, canonicalize_pq
"""

from .boundary import (
    csl_approx_spec_to_embedding,
    csl_spec_to_embedding,
    pq_spec_to_embedding,
    primitive_bicrystal_atom_count,
)
from .csl import (
    csl_from_scaled_rotation,
    dsc_basis,
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
    "csl_spec_to_embedding",
    "pq_spec_to_embedding",
    "primitive_bicrystal_atom_count",
    "CoincidenceCheck",
    "CSLResult",
    "DSCBasis",
    "InPlaneBasis",
    "ScaledRotation",
    "SmithDiagnostics",
    "normalize_integer_quaternion",
    "quaternion_to_scaled_rotation",
    "csl_from_scaled_rotation",
    "dsc_basis",
    "canonicalize_pq",
    "canonicalize_pq_paired",
    "recover_exact_row_rotation_from_paired_pq"
]
