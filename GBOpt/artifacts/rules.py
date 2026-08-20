# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define deterministic declarative rules for scientific archive membership.

Rules consume immutable :class:`RetentionCandidate` values and return logical candidate
identities representing the current archive membership for one criterion. This module
owns qualification, bounded ranking, and deterministic tie-breaking; it does not mutate
optimizer state, manage artifact references, or perform filesystem operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np

from GBOpt.artifacts.types import (
    ArtifactError,
    ArtifactValueError,
    RetentionCandidate,
    RetentionValue,
    normalize_retention_value,
    retention_value_to_state,
)

Direction = Literal["min", "max"]


class ArtifactRuleError(ArtifactError):
    """Raised when retention rule configuration or evaluation is invalid."""


class MissingRetentionPropertyError(ArtifactRuleError, KeyError):
    """Raised when an active retention rule requires a missing property."""


def _require_name(value: object, field_name: str) -> str:
    """Validate a non-empty rule/property identifier.

    :param value: Candidate identifier.
    :param field_name: Field name used in diagnostics.
    :return: Validated identifier.
    :raises ArtifactRuleError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ArtifactRuleError(f"{field_name} must be a non-empty string")
    return value


def _normalize_direction(value: object) -> Direction:
    """Validate common ranking direction vocabulary.

    :param value: Candidate ranking direction.
    :return: ``"min"`` or ``"max"``.
    :raises ArtifactRuleError: If the direction is unsupported.
    """
    if not isinstance(value, str) or value not in ("min", "max"):
        raise ArtifactRuleError("direction must be 'min' or 'max'")
    return value


def _normalize_bound(count: object, allow_unbounded: object) -> int | None:
    """Validate bounded-by-default retention cardinality.

    :param count: Candidate retention bound.
    :param allow_unbounded: Explicit opt-in to unbounded retention.
    :return: Positive Python integer or ``None`` for explicit unbounded retention.
    :raises ArtifactRuleError: If boundedness configuration is invalid.
    """
    if not isinstance(allow_unbounded, bool):
        raise ArtifactRuleError("allow_unbounded must be a bool")
    if count is None:
        if not allow_unbounded:
            raise ArtifactRuleError(
                "count is required unless allow_unbounded=True is explicit"
            )
        return None
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
        raise ArtifactRuleError("count must be a positive integer or None")
    normalized = int(count)
    if normalized <= 0:
        raise ArtifactRuleError("count must be a positive integer")
    return normalized


def _normalize_rank_configuration(
    *,
    count: int | None,
    rank_by: object,
    direction: object,
) -> tuple[str | None, Direction | None]:
    """Validate ranking fields for qualification-based rules.

    :param count: Keyword argument, required. Normalized retention bound.
    :param rank_by: Keyword argument, required. Candidate ranking property.
    :param direction: Keyword argument, required. Candidate ranking direction.
    :return: Normalized ranking property and direction.
    :raises ArtifactRuleError: If bounded rules omit ranking or unbounded rules supply
        it.
    """
    if count is None:
        if rank_by is not None or direction is not None:
            raise ArtifactRuleError(
                "unbounded qualification rules must not specify rank_by or direction"
            )
        return None, None
    if rank_by is None or direction is None:
        raise ArtifactRuleError("bounded rules require rank_by and direction")
    return _require_name(rank_by, "rank_by"), _normalize_direction(direction)


def _candidate_sequence(
    candidates: Iterable[RetentionCandidate],
) -> tuple[RetentionCandidate, ...]:
    """Validate candidate inputs and reject duplicate logical identities.

    :param candidates: Candidate population supplied to a rule.
    :return: Candidate population as a tuple.
    :raises ArtifactRuleError: If a value is not a retention candidate or an identity
        repeats.
    """
    try:
        result = tuple(candidates)
    except TypeError as exc:
        raise ArtifactRuleError(
            "rules require an iterable of RetentionCandidate values") from exc
    seen: set[str] = set()
    for candidate in result:
        if not isinstance(candidate, RetentionCandidate):
            raise ArtifactRuleError("rules require RetentionCandidate inputs")
        if candidate.candidate_id in seen:
            raise ArtifactRuleError(
                f"duplicate candidate identity {candidate.candidate_id!r}"
            )
        seen.add(candidate.candidate_id)
    return result


def _property(candidate: RetentionCandidate, name: str) -> RetentionValue:
    """Return a required candidate property with rule-specific diagnostics.

    :param candidate: Candidate containing the property namespace.
    :param name: Required property name.
    :return: Normalized property value.
    :raises MissingRetentionPropertyError: If the candidate lacks the required property.
    """
    try:
        return candidate.property_value(name)
    except KeyError as exc:
        raise MissingRetentionPropertyError(
            f"candidate {candidate.candidate_id!r} is missing required property "
            f"{name!r}"
        ) from exc


def _value_kind(value: RetentionValue) -> str:
    """Return the comparison family for one normalized retention value.

    :param value: Normalized retention value.
    :return: Stable comparison-family name.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    return "tuple"


def _ordering_key(value: RetentionValue) -> tuple:
    """Create a deterministic total-order key for supported retention values.

    :param value: Normalized retention value.
    :return: Recursive key preserving exact Python integer ranking.
    """
    if isinstance(value, bool):
        return (2, int(value))
    if isinstance(value, (int, float)):
        # Preserve Python integers exactly. Converting them to float can collapse
        # distinct values above 2**53 or overflow for arbitrary-sized integers.
        return (0, value)
    if isinstance(value, str):
        return (1, value)
    return (3, tuple(_ordering_key(item) for item in value))


def _ranked(
    candidates: Iterable[RetentionCandidate],
    *,
    property_name: str,
    direction: Direction,
) -> list[RetentionCandidate]:
    """Rank candidates with candidate-ID lexical tie-breaking in both directions.

    :param candidates: Candidates to rank.
    :param property_name: Keyword argument, required. Ranking property.
    :param direction: Keyword argument, required. ``"min"`` or ``"max"``.
    :return: Ranked candidate list.
    :raises MissingRetentionPropertyError: If a candidate lacks the ranking property.
    :raises ArtifactRuleError: If candidates use incompatible ranking value families.
    """
    result = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    values: dict[str, RetentionValue] = {}
    ranking_kind: str | None = None
    for candidate in result:
        value = _property(candidate, property_name)
        value_kind = _value_kind(value)
        if ranking_kind is None:
            ranking_kind = value_kind
        elif value_kind != ranking_kind:
            raise ArtifactRuleError(
                f"ranking property {property_name!r} must use compatible value types"
            )
        values[candidate.candidate_id] = value
    result.sort(
        key=lambda candidate: _ordering_key(values[candidate.candidate_id]),
        reverse=direction == "max",
    )
    return result


def _limit(
    candidates: Iterable[RetentionCandidate],
    *,
    count: int | None,
    rank_by: str | None,
    direction: Direction | None,
) -> tuple[str, ...]:
    """Apply optional bounded ranking to qualifying candidates.

    :param candidates: Qualifying candidates.
    :param count: Keyword argument, required. Retention bound or ``None``.
    :param rank_by: Keyword argument, required. Ranking property for bounded membership.
    :param direction: Keyword argument, required. Ranking direction for bounded
        membership.
    :return: Retained candidate identities in deterministic order.
    :raises MissingRetentionPropertyError: If bounded ranking requires a missing
        property.
    :raises ArtifactRuleError: If bounded ranking values use incompatible type families.
    """
    candidates = tuple(candidates)
    if count is None:
        return tuple(sorted(candidate.candidate_id for candidate in candidates))
    assert rank_by is not None
    assert direction is not None
    ranked = _ranked(candidates, property_name=rank_by, direction=direction)
    return tuple(candidate.candidate_id for candidate in ranked[:count])


def _is_discrete(value: RetentionValue) -> bool:
    """Return whether a value is suitable as a KeepDistinct bucket key.

    :param value: Normalized retention value.
    :return: Whether the value is categorical/discrete and contains no floats.
    """
    if isinstance(value, float):
        return False
    if isinstance(value, (bool, int, str)):
        return True
    return all(_is_discrete(item) for item in value)


class RetentionRule(ABC):
    """Abstract interface shared by declarative retention rules."""

    name: str

    @property
    @abstractmethod
    def required_properties(self) -> frozenset[str]:
        """Return properties whose presence can be validated declaratively"""

    @abstractmethod
    def select(
        self, candidates: Iterable[RetentionCandidate]
    ) -> tuple[str, ...]:
        """Return the current candidate IDs retained by this rule.

        :param candidates: Current evaluated candidate population.
        :return: Retained logical candidate identities.
        """

    @abstractmethod
    def to_state(self) -> dict[str, object]:
        """Return deterministic callback-free declarative rule state."""


@dataclass(frozen=True, slots=True, kw_only=True)
class KeepBest(RetentionRule):
    """Retain the current best candidates according to one property.

    :param name: Explicit persistent rule name.
    :param property: Property used directly for ranking.
    :param direction: ``"min"`` or ``"max"`` ranking direction.
    :param count: Positive retained count, or ``None`` with explicit unbounded opt-in.
    :param allow_unbounded: Explicit opt-in required when ``count`` is ``None``.
    :raises ArtifactRuleError: If identity, direction, or boundedness is invalid.
    """

    name: str
    property: str
    direction: Direction
    count: int | None
    allow_unbounded: bool = False

    def __post_init__(self) -> None:
        """Validate rule identity, ranking, and boundedness.

        :raises ArtifactRuleError: If rule configuration is invalid.
        """
        object.__setattr__(self, "name", _require_name(self.name, "name"))
        object.__setattr__(
            self, "property", _require_name(self.property, "property")
        )
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(
            self,
            "count",
            _normalize_bound(self.count, self.allow_unbounded),
        )

    @property
    def required_properties(self) -> frozenset[str]:
        """Return the ranking property required by this rule."""
        return frozenset({self.property})

    def select(
        self, candidates: Iterable[RetentionCandidate]
    ) -> tuple[str, ...]:
        """Return current best-N membership with deterministic ties.

        :param candidates: Current evaluated candidate population.
        :return: Retained candidate identities in ranking order.
        :raises ArtifactRuleError: If candidate input is malformed.
        :raises MissingRetentionPropertyError: If a ranking property is missing.
        """
        candidates = _candidate_sequence(candidates)
        ranked = _ranked(
            candidates, property_name=self.property, direction=self.direction
        )
        if self.count is not None:
            ranked = ranked[: self.count]
        return tuple(candidate.candidate_id for candidate in ranked)

    def to_state(self) -> dict[str, object]:
        """Return deterministic declarative rule state."""
        return {
            "type": "KeepBest",
            "name": self.name,
            "property": self.property,
            "direction": self.direction,
            "count": self.count,
            "allow_unbounded": self.allow_unbounded,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class KeepRange(RetentionRule):
    """Retain qualifying candidates inside an inclusive property range.

    :param name: Explicit persistent rule name.
    :param property: Property tested against the inclusive range.
    :param minimum: Inclusive lower bound.
    :param maximum: Inclusive upper bound.
    :param count: Positive retained count, or ``None`` with explicit unbounded opt-in.
    :param rank_by: Ranking property required for bounded membership.
    :param direction: Ranking direction required for bounded membership.
    :param allow_unbounded: Explicit opt-in required when ``count`` is ``None``.
    :raises ArtifactRuleError: If bounds, ranking, identity, or boundedness is invalid.
    """

    name: str
    property: str
    minimum: RetentionValue
    maximum: RetentionValue
    count: int | None = None
    rank_by: str | None = None
    direction: Direction | None = None
    allow_unbounded: bool = False
    _minimum_key: tuple = field(init=False, repr=False, compare=False)
    _maximum_key: tuple = field(init=False, repr=False, compare=False)
    _range_kind: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate inclusive bounds, ranking, and boundedness.

        :raises ArtifactRuleError: If bounds, ranking, or boundedness is invalid.
        """
        name = _require_name(self.name, "name")
        property_name = _require_name(self.property, "property")
        try:
            minimum = normalize_retention_value(self.minimum, name="minimum")
            maximum = normalize_retention_value(self.maximum, name="maximum")
        except ArtifactValueError as exc:
            raise ArtifactRuleError(str(exc)) from exc
        if _value_kind(minimum) != _value_kind(maximum):
            raise ArtifactRuleError(
                "minimum and maximum must use comparable value types")
        minimum_key = _ordering_key(minimum)
        maximum_key = _ordering_key(maximum)
        if minimum_key > maximum_key:
            raise ArtifactRuleError("minimum must be less than or equal to maximum")
        count = _normalize_bound(self.count, self.allow_unbounded)
        rank_by, direction = _normalize_rank_configuration(
            count=count, rank_by=self.rank_by, direction=self.direction
        )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "property", property_name)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "rank_by", rank_by)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "_minimum_key", minimum_key)
        object.__setattr__(self, "_maximum_key", maximum_key)
        object.__setattr__(self, "_range_kind", _value_kind(minimum))

    @property
    def required_properties(self) -> frozenset[str]:
        """Return qualification and ranking properties.

        :return: Required property-name set.
        """
        result = {self.property}
        if self.rank_by is not None:
            result.add(self.rank_by)
        return frozenset(result)

    def select(
        self, candidates: Iterable[RetentionCandidate]
    ) -> tuple[str, ...]:
        """Return current bounded membership within the inclusive range.

        :param candidates: Current evaluated candidate population.
        :return: Retained candidate identities.
        :raises ArtifactRuleError: If candidate values are incompatible with the range.
        :raises MissingRetentionPropertyError: If a required property is missing.
        """
        candidates = _candidate_sequence(candidates)
        qualifying: list[RetentionCandidate] = []
        for candidate in candidates:
            value = _property(candidate, self.property)
            if _value_kind(value) != self._range_kind:
                raise ArtifactRuleError(
                    f"candidate {candidate.candidate_id!r} property {self.property!r} "
                    "has a value type incompatible with the configured range"
                )
            key = _ordering_key(value)
            if self._minimum_key <= key <= self._maximum_key:
                qualifying.append(candidate)
        return _limit(
            qualifying,
            count=self.count,
            rank_by=self.rank_by,
            direction=self.direction,
        )

    def to_state(self) -> dict[str, object]:
        """Return deterministic declarative rule state."""
        return {
            "type": "KeepRange",
            "name": self.name,
            "property": self.property,
            "minimum": retention_value_to_state(self.minimum),
            "maximum": retention_value_to_state(self.maximum),
            "count": self.count,
            "rank_by": self.rank_by,
            "direction": self.direction,
            "allow_unbounded": self.allow_unbounded,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class KeepDistinct(RetentionRule):
    """Retain bounded representatives for each discrete property value.

    :param name: Explicit persistent rule name.
    :param property: Categorical property used to form buckets.
    :param per_value: Positive retained count for each encountered discrete value.
    :param rank_by: Property used to rank candidates within each bucket.
    :param direction: ``"min"`` or ``"max"`` ranking direction.
    :raises ArtifactRuleError: If configuration is malformed.
    """

    name: str
    property: str
    per_value: int
    rank_by: str
    direction: Direction

    def __post_init__(self) -> None:
        """Validate categorical grouping and ranking configuration.

        :raises ArtifactRuleError: If grouping or ranking configuration is invalid.
        """
        object.__setattr__(self, "name", _require_name(self.name, "name"))
        object.__setattr__(
            self, "property", _require_name(self.property, "property")
        )
        if isinstance(self.per_value, (bool, np.bool_)) or not isinstance(
            self.per_value, Integral
        ):
            raise ArtifactRuleError("per_value must be a positive integer")
        per_value = int(self.per_value)
        if per_value <= 0:
            raise ArtifactRuleError("per_value must be a positive integer")
        object.__setattr__(self, "per_value", per_value)
        object.__setattr__(self, "rank_by", _require_name(self.rank_by, "rank_by"))
        object.__setattr__(self, "direction", _normalize_direction(self.direction))

    @property
    def required_properties(self) -> frozenset[str]:
        """Return grouping and ranking properties."""
        return frozenset({self.property, self.rank_by})

    def select(self, candidates: Iterable[RetentionCandidate]) -> tuple[str, ...]:
        """Return bounded representatives from each discrete value bucket.

        :param candidates: Current evaluated candidate population.
        :return: Retained candidate identities grouped in deterministic bucket order.
        :raises ArtifactRuleError: If a grouping value is not discrete.
        :raises MissingRetentionPropertyError: If grouping or ranking properties are
            missing.
        """
        candidates = _candidate_sequence(candidates)
        buckets: dict[RetentionValue, list[RetentionCandidate]] = defaultdict(list)
        for candidate in candidates:
            value = _property(candidate, self.property)
            if not _is_discrete(value):
                raise ArtifactRuleError(
                    f"KeepDistinct property {self.property!r} must be categorical; "
                    "floating-point values are not allowed"
                )
            buckets[value].append(candidate)

        selected: list[str] = []
        for value in sorted(buckets, key=_ordering_key):
            ranked = _ranked(
                buckets[value], property_name=self.rank_by, direction=self.direction
            )
            selected.extend(
                candidate.candidate_id for candidate in ranked[: self.per_value]
            )
        return tuple(selected)

    def to_state(self) -> dict[str, object]:
        """Return deterministic declarative rule state."""
        return {
            "type": "KeepDistinct",
            "name": self.name,
            "property": self.property,
            "per_value": self.per_value,
            "rank_by": self.rank_by,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class KeepIf(RetentionRule):
    """Retain bounded candidates that satisfy a stateless predicate.

    :param name: Explicit persistent rule name.
    :param predicate: Candidate-local Boolean qualification callback.
    :param version: Explicit callback contract/configuration version.
    :param count: Positive retained count, or ``None`` with explicit unbounded opt-in.
    :param rank_by: Ranking property required for bounded membership.
    :param direction: Ranking direction required for bounded membership.
    :param allow_unbounded: Explicit opt-in required when ``count`` is ``None``.
    :raises ArtifactRuleError: If callback, identity, ranking, or boundedness is
        invalid.
    """

    name: str
    predicate: Callable[[RetentionCandidate], object] = field(compare=False, repr=False)
    version: str
    count: int | None = None
    rank_by: str | None = None
    direction: Direction | None = None
    allow_unbounded: bool = False

    def __post_init__(self) -> None:
        """Validate callback identity, ranking, and boundedness.

        :raises ArtifactRuleError: If callback, identity, ranking, or boundedness is
            invalid.
        """
        name = _require_name(self.name, "name")
        if not callable(self.predicate):
            raise ArtifactRuleError("predicate must be callable")
        version = _require_name(self.version, "version")
        count = _normalize_bound(self.count, self.allow_unbounded)
        rank_by, direction = _normalize_rank_configuration(
            count=count, rank_by=self.rank_by, direction=self.direction
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "rank_by", rank_by)
        object.__setattr__(self, "direction", direction)

    @property
    def required_properties(self) -> frozenset[str]:
        """Return declaratively known ranking properties.

        Predicate dependencies are intentionally opaque. A missing property accessed by
        the predicate is converted to :class:`MissingRetentionPropertyError` at runtime.

        :return: Declaratively known ranking properties only.
        """
        if self.rank_by is None:
            return frozenset()
        return frozenset({self.rank_by})

    def select(
        self, candidates: Iterable[RetentionCandidate]
    ) -> tuple[str, ...]:
        """Return candidates whose predicate qualifies them, then apply ranking.

        :param candidates: Current evaluated candidate population.
        :return: Retained candidate identities.
        :raises ArtifactRuleError: If the predicate does not return a Boolean value.
        :raises MissingRetentionPropertyError: If the predicate or ranking needs a
            missing property.
        """
        candidates = _candidate_sequence(candidates)
        qualifying: list[RetentionCandidate] = []
        for candidate in candidates:
            try:
                result = self.predicate(candidate)
            except KeyError as exc:
                missing = exc.args[0] if exc.args else None
                if isinstance(missing, str) and missing not in candidate.properties:
                    raise MissingRetentionPropertyError(
                        f"candidate {candidate.candidate_id!r} is missing property "
                        f"{missing!r} required by predicate {self.name!r}"
                    ) from exc
                raise
            if not isinstance(result, (bool, np.bool_)):
                raise ArtifactRuleError(
                    f"predicate {self.name!r} must return a Boolean value"
                )
            if bool(result):
                qualifying.append(candidate)
        return _limit(
            qualifying,
            count=self.count,
            rank_by=self.rank_by,
            direction=self.direction,
        )

    def to_state(self) -> dict[str, object]:
        """Return callback-free declarative rule state with explicit version."""
        return {
            "type": "KeepIf",
            "name": self.name,
            "version": self.version,
            "count": self.count,
            "rank_by": self.rank_by,
            "direction": self.direction,
            "allow_unbounded": self.allow_unbounded,
        }


__all__ = [
    "ArtifactRuleError",
    "KeepBest",
    "KeepDistinct",
    "KeepIf",
    "KeepRange",
    "MissingRetentionPropertyError",
]
