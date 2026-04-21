# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings

import pytest


def pytest_runtest_makereport(item, call):
    if "known_bug" in item.keywords:
        if call.excinfo is None:
            warnings.warn(
                f"Test {item.name} passed but is marked as a known bug", UserWarning)
        elif call.excinfo.typename != "AssertionError":
            warnings.warn(
                f"Test {item.name} failed due to an unexpected error: {call.excinfo.value}", UserWarning)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        print(f"\n[WARNING in {item.nodeid}] {w.category.__name__}: {w.message}")
