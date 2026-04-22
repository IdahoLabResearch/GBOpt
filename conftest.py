# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings


def pytest_runtest_makereport(item, call):
    if "known_bug" in item.keywords:
        if call.excinfo is None:
            warnings.warn(
                f"Test {item.name} passed but is marked as a known bug", UserWarning)
        elif call.excinfo.typename != "AssertionError":
            warnings.warn(
                f"Test {item.name} failed due to an unexpected error: {call.excinfo.value}", UserWarning)


def pytest_warning_recorded(warning_message, when, nodeid, location):
    print(
        f"\n[WARNING in {nodeid}] {warning_message.category.__name__}: {warning_message.message}")
