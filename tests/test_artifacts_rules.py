# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json

import numpy as np
import pytest

from GBOpt.artifacts.rules import (
    ArtifactRuleError,
    KeepBest,
    KeepDistinct,
    KeepIf,
    KeepRange,
    MissingRetentionPropertyError,
)
from GBOpt.artifacts.types import RetentionCandidate


def _candidate(candidate_id, objective, **properties):
    return RetentionCandidate(
        candidate_id=candidate_id,
        generation=0,
        objective=objective,
        properties=properties,
    )


def test_keep_best_selects_best_n_minimum():
    rule = KeepBest(
        name="objective_elite", property="objective", direction="min", count=2
    )
    candidates = [
        _candidate("c", 3.0),
        _candidate("a", 1.0),
        _candidate("b", 2.0),
    ]

    assert rule.select(candidates) == ("a", "b")


def test_keep_best_selects_best_n_maximum():
    rule = KeepBest(name="high_density", property="density", direction="max", count=2)
    candidates = [
        _candidate("a", 1.0, density=10.0),
        _candidate("b", 2.0, density=12.0),
        _candidate("c", 3.0, density=11.0),
    ]

    assert rule.select(candidates) == ("b", "c")


def test_ranking_ties_use_candidate_id_lexical_order_in_both_directions():
    candidates = [_candidate("c", 1.0), _candidate("a", 1.0), _candidate("b", 1.0)]

    minimum = KeepBest(name="min", property="objective", direction="min", count=3)
    maximum = KeepBest(name="max", property="objective", direction="max", count=3)

    assert minimum.select(candidates) == ("a", "b", "c")
    assert maximum.select(candidates) == ("a", "b", "c")


def test_numeric_ranking_preserves_arbitrary_size_python_integers():
    rule = KeepBest(name="largest", property="rank", direction="max", count=2)
    candidates = [
        _candidate("smaller", 0.0, rank=2**53),
        _candidate("larger", 0.0, rank=2**53 + 1),
        _candidate("huge", 0.0, rank=10**400),
    ]

    assert rule.select(candidates) == ("huge", "larger")


def test_ranking_rejects_mixed_property_value_families():
    rule = KeepBest(name="rank", property="rank", direction="min", count=2)

    with pytest.raises(ArtifactRuleError, match="compatible value types"):
        rule.select(
            [
                _candidate("numeric", 0.0, rank=1),
                _candidate("text", 0.0, rank="1"),
            ]
        )


def test_keep_range_uses_inclusive_bounds():
    rule = KeepRange(
        name="target",
        property="density",
        minimum=10.5,
        maximum=11.0,
        count=3,
        rank_by="objective",
        direction="min",
    )
    candidates = [
        _candidate("low", 0.0, density=10.49),
        _candidate("minimum", 3.0, density=10.5),
        _candidate("middle", 2.0, density=10.7),
        _candidate("maximum", 1.0, density=11.0),
        _candidate("high", 0.0, density=11.01),
    ]

    assert rule.select(candidates) == ("maximum", "middle", "minimum")


def test_keep_range_bounded_membership_replaces_worse_candidate():
    rule = KeepRange(
        name="target",
        property="density",
        minimum=10.0,
        maximum=11.0,
        count=2,
        rank_by="objective",
        direction="min",
    )
    first = [_candidate("a", 3.0, density=10.5), _candidate("b", 2.0, density=10.5)]
    later = [*first, _candidate("c", 1.0, density=10.5)]

    assert rule.select(first) == ("b", "a")
    assert rule.select(later) == ("c", "b")


def test_keep_distinct_keeps_bounded_representatives_per_categorical_value():
    rule = KeepDistinct(
        name="composition_archive",
        property="composition",
        per_value=1,
        rank_by="objective",
        direction="min",
    )
    candidates = [
        _candidate("a2", 2.0, composition=(("A", 1),)),
        _candidate("b1", 1.0, composition=(("B", 1),)),
        _candidate("a1", 1.0, composition=(("A", 1),)),
        _candidate("b2", 2.0, composition=(("B", 1),)),
    ]

    assert rule.select(candidates) == ("a1", "b1")


def test_keep_distinct_rejects_floating_point_bucket_keys():
    rule = KeepDistinct(
        name="bad_density_distinct",
        property="density",
        per_value=1,
        rank_by="objective",
        direction="min",
    )

    with pytest.raises(ArtifactRuleError, match="floating-point"):
        rule.select([_candidate("a", 1.0, density=10.9)])


def test_keep_if_qualifies_locally_then_ranks_bounded_membership():
    rule = KeepIf(
        name="oxygen_deficient",
        predicate=lambda candidate: candidate.properties["excess_o"] < 0,
        version="1",
        count=2,
        rank_by="objective",
        direction="min",
    )
    candidates = [
        _candidate("a", 3.0, excess_o=-1.0),
        _candidate("b", 2.0, excess_o=1.0),
        _candidate("c", 1.0, excess_o=-2.0),
        _candidate("d", 2.0, excess_o=-3.0),
    ]

    assert rule.select(candidates) == ("c", "d")


def test_keep_if_requires_explicit_callback_version():
    with pytest.raises(ArtifactRuleError, match="version"):
        KeepIf(
            name="custom",
            predicate=lambda candidate: True,
            version="",
            count=1,
            rank_by="objective",
            direction="min",
        )


def test_keep_if_state_excludes_callback_object_and_includes_version():
    rule = KeepIf(
        name="custom",
        predicate=lambda candidate: True,
        version="v2",
        count=1,
        rank_by="objective",
        direction="min",
    )

    state = rule.to_state()

    assert state["version"] == "v2"
    assert "predicate" not in state
    json.dumps(state, sort_keys=True)


def test_rule_count_rejects_boolean_values():
    with pytest.raises(ArtifactRuleError, match="positive integer"):
        KeepBest(
            name="elite",
            property="objective",
            direction="min",
            count=True,
        )


def test_rules_are_bounded_by_default():
    with pytest.raises(ArtifactRuleError, match="allow_unbounded"):
        KeepIf(name="all", predicate=lambda candidate: True, version="1")


def test_explicit_unbounded_opt_in_is_supported():
    rule = KeepIf(
        name="all",
        predicate=lambda candidate: True,
        version="1",
        allow_unbounded=True,
    )

    assert rule.select([_candidate("b", 2.0), _candidate("a", 1.0)]) == ("a", "b")
    assert rule.to_state()["allow_unbounded"] is True


def test_unbounded_qualification_rules_reject_unused_ranking_configuration():
    with pytest.raises(ArtifactRuleError, match="must not specify"):
        KeepIf(
            name="all",
            predicate=lambda candidate: True,
            version="1",
            allow_unbounded=True,
            rank_by="objective",
            direction="min",
        )


def test_bounded_qualification_rules_require_explicit_ranking_semantics():
    with pytest.raises(ArtifactRuleError, match="rank_by and direction"):
        KeepRange(
            name="target",
            property="density",
            minimum=10.0,
            maximum=11.0,
            count=2,
        )


def test_missing_required_property_fails_explicitly():
    rule = KeepBest(name="density", property="density", direction="max", count=1)

    with pytest.raises(MissingRetentionPropertyError, match="density"):
        rule.select([_candidate("a", 1.0)])


def test_missing_predicate_property_fails_explicitly():
    rule = KeepIf(
        name="custom",
        predicate=lambda candidate: candidate.properties["missing"] == 1,
        version="1",
        count=1,
        rank_by="objective",
        direction="min",
    )

    with pytest.raises(MissingRetentionPropertyError, match="missing"):
        rule.select([_candidate("a", 1.0)])


def test_non_boolean_predicate_result_is_rejected():
    rule = KeepIf(
        name="custom",
        predicate=lambda candidate: 1,
        version="1",
        count=1,
        rank_by="objective",
        direction="min",
    )

    with pytest.raises(ArtifactRuleError, match="Boolean"):
        rule.select([_candidate("a", 1.0)])


def test_rule_selection_is_independent_of_input_order():
    rule = KeepBest(name="elite", property="objective", direction="min", count=2)
    candidates = [_candidate("a", 1.0), _candidate("b", 1.0), _candidate("c", 2.0)]

    assert rule.select(candidates) == rule.select(list(reversed(candidates)))


def test_numpy_boolean_predicate_result_is_normalized():
    rule = KeepIf(
        name="numpy_bool",
        predicate=lambda candidate: np.bool_(candidate.objective < 2.0),
        version="1",
        count=1,
        rank_by="objective",
        direction="min",
    )

    assert rule.select([_candidate("a", 1.0)]) == ("a",)
