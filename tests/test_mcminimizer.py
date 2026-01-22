# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math
import unittest
import uuid

import numpy as np

from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import MonteCarloMinimizer


class TestMonteCarloMinimizer(unittest.TestCase):
    def setUp(self):
        theta = math.radians(36.869898)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gb = GBMaker(
            3.52,
            "fcc",
            10.0,
            misorientation,
            "Ni",
            repeat_factor=2,
            x_dim_min=30.0,
            vacuum=8.0,
            interaction_distance=8.0,
        )

    def test_run_mc_default_unique_id_is_per_call(self):
        observed_ids = []

        def fake_energy_func(_gb, _manipulator, _atom_positions, unique_id):
            observed_ids.append(unique_id)
            return 0.0, "dummy.dump"

        minimizer = MonteCarloMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
        )

        minimizer.run_MC(max_steps=0)
        minimizer.run_MC(max_steps=0)

        self.assertEqual(len(observed_ids), 2)
        first = observed_ids[0]
        second = observed_ids[1]

        self.assertTrue(first.startswith("initial"))
        self.assertTrue(second.startswith("initial"))
        first_uuid = uuid.UUID(first[len("initial"):])
        second_uuid = uuid.UUID(second[len("initial"):])
        self.assertNotEqual(first_uuid, second_uuid)


if __name__ == "__main__":
    unittest.main()
