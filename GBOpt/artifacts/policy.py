# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define retention-policy configuration and candidate-property acquisition.

This module combines declarative scientific rules with optional application property
providers, validates their persistent identity, and produces deterministic callback-free
policy state. It consumes validated :class:`CandidatePropertyContext` values and returns
narrow :class:`RetentionCandidate` values. Runtime artifact references and filesystem
lifecycle operations belong to the artifact store and later cleanup layers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from GBOpt.artifacts.rules import ArtifactRuleError, RetentionRule
from GBOpt.artifacts.types import (
    ArtifactError,
    ArtifactValueError,
    CandidatePropertyContext,
    RetentionCandidate,
    RetentionValue,
    normalize_property_mapping,
    retention_value_to_state,
)

_POLICY_VERSION = 1

__all__ = ["ArtifactPolicyError", "ArtifactRetentionPolicy"]


class ArtifactPolicyError(ArtifactError):
    """Raised when retention policy configuration or property acquisition is invalid."""


def _require_nonempty_string(value: object, name: str) -> str:
    """Validate a non-empty string policy field.

    :param value: Candidate string value.
    :param name: Field name used in diagnostics.
    :return: Validated string.
    :raises ArtifactPolicyError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ArtifactPolicyError(f"{name} must be a non-empty string")
    return value


def _provider_identity(provider: Callable, explicit_name: str | None) -> str:
    """Return a stable declarative identity for a property-provider callback.

    :param provider: Runtime property-provider callback.
    :param explicit_name: Optional stable identity supplied by the application.
    :return: Persistent callback identity included in policy state.
    :raises ArtifactPolicyError: If no stable callback identity can be established.
    """
    if explicit_name is not None:
        return _require_nonempty_string(explicit_name, "property_provider_name")
    module = getattr(provider, "__module__", None)
    qualname = getattr(provider, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(qualname, str)
        or not qualname
        or "<lambda>" in qualname
    ):
        raise ArtifactPolicyError(
            "property_provider_name is required when callback identity cannot be derived "
            "uniquely"
        )
    return f"{module}.{qualname}"


def _builtin_properties(context: CandidatePropertyContext) -> dict[str, RetentionValue]:
    """Calculate guaranteed physical built-in retention properties.

    :param context: Validated relaxed physical candidate state.
    :return: Built-in atom-count, composition, and cell-volume properties.
    :raises ArtifactPolicyError: If the derived cell volume is not finite and positive.
    """
    species_counts = Counter(str(name) for name in context.atoms["name"])
    composition = tuple(sorted(species_counts.items()))
    lengths = context.box_dims[:, 1] - context.box_dims[:, 0]
    cell_volume = float(np.prod(lengths))
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ArtifactPolicyError("candidate cell_volume must be finite and positive")
    return {
        "atom_count": len(context.atoms),
        "composition": composition,
        "cell_volume": cell_volume,
    }


@dataclass(frozen=True, slots=True)
class ArtifactRetentionPolicy:
    """Reusable retention-rule and candidate-property configuration.

    :param rules: Declarative scientific retention rules. Rule names must be unique.
    :param property_provider: Optional callback receiving validated relaxed physical
        candidate state and returning simple named properties.
    :param property_provider_version: Explicit callback implementation/configuration
        version used for checkpoint compatibility.
    :param property_provider_name: Optional stable callback identity override. When
        omitted, ``module.qualname`` is used.
    :param property_provider_config: Optional declarative callback configuration included
        in the deterministic policy signature.
    :param prune: Whether unreferenced candidates may eventually be pruned. Stage A does
        not perform filesystem deletion.
    :raises ArtifactPolicyError: If rule, provider, configuration, or pruning settings
        violate the policy contract.
    """

    rules: tuple[RetentionRule, ...] = ()
    property_provider: Callable[[CandidatePropertyContext], Mapping[object, object]] | None = field(
        default=None, repr=False, compare=False
    )
    property_provider_version: str | None = None
    property_provider_name: str | None = None
    property_provider_config: Mapping[str, RetentionValue] = field(default_factory=dict)
    prune: bool = False
    _provider_identity: str | None = field(init=False, repr=False, compare=False)
    _signature: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate rules, callback identity, configuration, and deterministic state.

        :raises ArtifactPolicyError: If policy configuration is malformed.
        """
        try:
            rules = tuple(self.rules)
        except TypeError as exc:
            raise ArtifactPolicyError("rules must be an iterable of RetentionRule values") from exc
        for rule in rules:
            if not isinstance(rule, RetentionRule):
                raise ArtifactPolicyError("rules must contain only RetentionRule values")
        names = [
            _require_nonempty_string(rule.name, "rule name")
            for rule in rules
        ]
        duplicates = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        if duplicates:
            raise ArtifactPolicyError(
                "retention rule names must be unique; duplicates: " + ", ".join(duplicates)
            )
        rules = tuple(sorted(rules, key=lambda rule: rule.name))

        if type(self.prune) is not bool:
            raise ArtifactPolicyError("prune must be a bool")

        provider = self.property_provider
        provider_identity: str | None = None
        provider_version = self.property_provider_version
        if provider is None:
            if provider_version is not None or self.property_provider_name is not None:
                raise ArtifactPolicyError(
                    "property provider identity/version requires property_provider"
                )
            if self.property_provider_config:
                raise ArtifactPolicyError(
                    "property_provider_config requires property_provider"
                )
        else:
            if not callable(provider):
                raise ArtifactPolicyError("property_provider must be callable or None")
            provider_version = _require_nonempty_string(
                provider_version, "property_provider_version"
            )
            provider_identity = _provider_identity(provider, self.property_provider_name)

        try:
            provider_config = normalize_property_mapping(self.property_provider_config)
        except ArtifactValueError as exc:
            raise ArtifactPolicyError(
                f"property_provider_config is invalid: {exc}"
            ) from exc

        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "property_provider_version", provider_version)
        object.__setattr__(self, "property_provider_config", provider_config)
        object.__setattr__(self, "_provider_identity", provider_identity)
        state = self.to_state()
        encoded = json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        object.__setattr__(self, "_signature", hashlib.sha256(encoded).hexdigest())

    @classmethod
    def keep_all(cls) -> "ArtifactRetentionPolicy":
        """Return the explicit keep-all/default-compatible policy.

        :return: Non-pruning policy with no scientific rules.
        """
        return cls(rules=(), prune=False)

    @property
    def signature(self) -> str:
        """Return the deterministic callback-free policy signature.

        :return: SHA-256 hexadecimal digest of declarative policy state.
        """
        return self._signature

    @property
    def rule_names(self) -> tuple[str, ...]:
        """Return rule names in deterministic lexical order.

        :return: Persistent rule names.
        """
        return tuple(rule.name for rule in self.rules)

    @property
    def required_properties(self) -> frozenset[str]:
        """Return declaratively known rule property dependencies.

        :return: Union of property names required by configured rules.
        """
        required: set[str] = set()
        for rule in self.rules:
            required.update(rule.required_properties)
        return frozenset(required)

    def candidate_from_context(
        self,
        context: CandidatePropertyContext,
        *,
        lineage: tuple[str, ...] = (),
    ) -> RetentionCandidate:
        """Acquire built-in/user properties and create the narrow rule candidate view.

        :param context: Validated relaxed physical candidate state.
        :param lineage: Keyword argument, optional, defaults to ``()``. Stable logical parent
            identities.
        :return: Immutable retention candidate with normalized properties.
        :raises ArtifactPolicyError: If provider output is malformed, collides with the
            reserved built-in namespace, or lacks a property needed by an active rule.
        """
        if not isinstance(context, CandidatePropertyContext):
            raise ArtifactPolicyError("context must be a CandidatePropertyContext")
        properties = _builtin_properties(context)
        if self.property_provider is not None:
            raw_user_properties = self.property_provider(context)
            try:
                user_properties = normalize_property_mapping(
                    raw_user_properties, reject_reserved=True
                )
            except ArtifactValueError as exc:
                raise ArtifactPolicyError(
                    f"property provider output is invalid: {exc}"
                ) from exc
            properties.update(user_properties)

        try:
            candidate = RetentionCandidate(
                candidate_id=context.candidate_id,
                generation=context.generation,
                objective=context.objective,
                properties=properties,
                lineage=lineage,
            )
        except ArtifactValueError as exc:
            raise ArtifactPolicyError(f"retention candidate state is invalid: {exc}") from exc
        missing = sorted(
            property_name
            for property_name in self.required_properties
            if property_name not in candidate.properties
        )
        if missing:
            raise ArtifactPolicyError(
                f"candidate {candidate.candidate_id!r} is missing properties required "
                f"by active retention rules: {', '.join(missing)}"
            )
        return candidate

    def evaluate(
        self, candidates: tuple[RetentionCandidate, ...] | list[RetentionCandidate]
    ) -> dict[str, tuple[str, ...]]:
        """Evaluate every scientific rule against the same candidate population.

        :param candidates: Current set of evaluated scientific candidates.
        :return: Rule-name to current retained candidate IDs.
        :raises ArtifactPolicyError: If the population container is invalid or a configured
            rule cannot evaluate it.
        """
        if not isinstance(candidates, (tuple, list)):
            raise ArtifactPolicyError(
                "candidates must be a tuple or list of RetentionCandidate values"
            )
        population = tuple(candidates)
        memberships: dict[str, tuple[str, ...]] = {}
        for rule in self.rules:
            try:
                memberships[rule.name] = rule.select(population)
            except ArtifactRuleError as exc:
                raise ArtifactPolicyError(
                    f"retention rule {rule.name!r} could not evaluate candidates: {exc}"
                ) from exc
        return memberships

    def to_state(self) -> dict[str, object]:
        """Return deterministic JSON-safe policy state without callback objects.

        :return: Declarative policy state used for signatures and checkpoint compatibility.
        """
        provider_state: dict[str, object] | None
        if self.property_provider is None:
            provider_state = None
        else:
            provider_state = {
                "name": self._provider_identity,
                "version": self.property_provider_version,
                "config": {
                    key: retention_value_to_state(value)
                    for key, value in self.property_provider_config.items()
                },
            }
        return {
            "version": _POLICY_VERSION,
            "prune": self.prune,
            "rules": [rule.to_state() for rule in self.rules],
            "property_provider": provider_state,
        }
