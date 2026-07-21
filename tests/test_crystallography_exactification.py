# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for crystallographic exactification operations."""

import numpy as np
import pytest

from GBOpt.crystallography.exactification import exactify_five_dof


def test_exactify_five_dof_is_not_implemented():
    params = np.zeros(5, dtype=float)

    with pytest.raises(
        NotImplementedError,
        match=r"five-DOF parameters to exact canonical P/Q matrices",
    ):
        exactify_five_dof(params)
