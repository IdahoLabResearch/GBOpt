# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the supported scientific artifact-retention API.

The package-level surface contains retention policy configuration, built-in scientific
rules, callback-facing candidate values, and safe evaluator-cleanup primitives. Runtime
store internals, operational pin categories, reserved-name tables, and serialization
helpers remain available from their defining modules for GBOpt integration but are not
promoted as user-facing API.
"""

from .cleanup import ArtifactCleanupError, ArtifactCleanupRequest, remove_managed_path
from .policy import ArtifactPolicyError, ArtifactRetentionPolicy
from .rules import (
    ArtifactRuleError,
    KeepBest,
    KeepDistinct,
    KeepIf,
    KeepRange,
    MissingRetentionPropertyError,
)
from .types import (
    ArtifactError,
    ArtifactValueError,
    CandidatePropertyContext,
    RetentionCandidate,
    RetentionValue,
)

__all__ = [
    # Exceptions
    "ArtifactError",
    "ArtifactValueError",
    "ArtifactRuleError",
    "MissingRetentionPropertyError",
    "ArtifactPolicyError",
    "ArtifactCleanupError",
    # Callback-facing domain values
    "RetentionValue",
    "RetentionCandidate",
    "CandidatePropertyContext",
    "ArtifactCleanupRequest",
    # Scientific rules and policy
    "KeepBest",
    "KeepRange",
    "KeepDistinct",
    "KeepIf",
    "ArtifactRetentionPolicy",
    # Safe cleanup primitive
    "remove_managed_path",
]
