# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for low-level boundary-normal topology metadata."""

import pytest

from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    BoundaryTopologyError,
    normalize_boundary_normal_topology,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, BoundaryNormalTopology.UNKNOWN, id="missing"),
        pytest.param(
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            id="periodic-enum",
        ),
        pytest.param(
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            id="slab-enum",
        ),
        pytest.param(
            "periodic_bicrystal",
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            id="periodic-string",
        ),
        pytest.param(
            "single_interface_slab",
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            id="slab-string",
        ),
    ],
)
def test_normalize_boundary_normal_topology(value, expected):
    assert normalize_boundary_normal_topology(value) is expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("periodic", id="unsupported-string"),
        pytest.param(True, id="boolean"),
        pytest.param(1, id="integer"),
        pytest.param([], id="list"),
    ],
)
def test_normalize_boundary_normal_topology_rejects_invalid_values(value):
    with pytest.raises(
        BoundaryTopologyError,
        match="Unsupported boundary-normal topology",
    ):
        normalize_boundary_normal_topology(value)


def test_periodic_outer_interface_is_derived_from_topology():
    assert BoundaryNormalTopology.PERIODIC_BICRYSTAL.periodic_outer_x_interface
    assert not BoundaryNormalTopology.SINGLE_INTERFACE_SLAB.periodic_outer_x_interface
    assert not BoundaryNormalTopology.UNKNOWN.periodic_outer_x_interface
