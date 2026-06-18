# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings

import pytest


def pytest_runtest_makereport(item, call):
    if "known_bug" in item.keywords and call.when == "call":
        if call.excinfo is None:
            warnings.warn(
                f"Test {item.name} passed but is marked as a known bug", UserWarning)
        elif call.excinfo.typename != "AssertionError":
            warnings.warn(
                f"Test {item.name} failed due to an unexpected error: {call.excinfo.value}", UserWarning)


def pytest_warning_recorded(warning_message, when, nodeid, location):
    print(
        f"\n[WARNING in {nodeid}] {warning_message.category.__name__}: {warning_message.message}")

# ---------------------------------------------------------------------------
# Crystallography fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sigma5_53deg_rotation():
    """Sigma5 [001] 53.13 deg scaled rotation -- quaternion (2, 0, 0, 1), N=5."""
    from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
    return quaternion_to_scaled_rotation((2, 0, 0, 1))


@pytest.fixture
def sigma5_36deg_rotation():
    """Sigma5 [001] 36.87 deg scaled rotation -- quaternion (3, 0, 0, 1), N=10."""
    from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
    return quaternion_to_scaled_rotation((3, 0, 0, 1))


@pytest.fixture
def sigma3_111_rotation():
    """Sigma3 [111] 60 deg twin scaled rotation -- quaternion (1, 1, 1, 0), N=3."""
    from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
    return quaternion_to_scaled_rotation((1, 1, 1, 0))
