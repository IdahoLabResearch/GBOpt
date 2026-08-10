# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pytest

from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    BoundaryTopologyError,
    normalize_boundary_normal_topology,
)

# ---------------------------------------------------------------------------
# normalize_boundary_normal_topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            None,
            BoundaryNormalTopology.UNKNOWN,
            id="missing",
        ),
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
            BoundaryNormalTopology.UNKNOWN,
            BoundaryNormalTopology.UNKNOWN,
            id="unknown-enum",
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
        pytest.param(
            "unknown",
            BoundaryNormalTopology.UNKNOWN,
            id="unknown-string",
        ),
    ],
)
def test_normalize_boundary_normal_topology_accepts_supported_values(value, expected):
    assert normalize_boundary_normal_topology(value) is expected


@pytest.mark.parametrize(
    ("value", "match"),
    [
        pytest.param(
            "periodic",
            r"Unsupported boundary-normal topology: 'periodic'",
            id="unsupported-string",
        ),
        pytest.param(
            True,
            r"Unsupported boundary-normal topology: True",
            id="boolean",
        ),
        pytest.param(
            1,
            r"Unsupported boundary-normal topology: 1",
            id="integer",
        ),
        pytest.param(
            [],
            r"Unsupported boundary-normal topology: \[\]",
            id="list",
        ),
    ],
)
def test_normalize_boundary_normal_topology_rejects_invalid_values(value, match):
    with pytest.raises(BoundaryTopologyError, match=match):
        normalize_boundary_normal_topology(value)


# ---------------------------------------------------------------------------
# BoundaryNormalTopology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("topology", "expected"),
    [
        pytest.param(
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            True,
            id="periodic-bicrystal",
        ),
        pytest.param(
            BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
            False,
            id="single-interface-slab",
        ),
        pytest.param(
            BoundaryNormalTopology.UNKNOWN,
            False,
            id="unknown",
        ),
    ],
)
def test_periodic_outer_x_interface_reflects_boundary_topology(topology, expected):
    assert topology.periodic_outer_x_interface is expected
