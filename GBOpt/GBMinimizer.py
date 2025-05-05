import numpy as np
from GBOpt import GBMaker, GBManipulator
import math
import uuid
from time import time
import sys
import shutil


class Mutator:
    """
    Mutator class for performing random manipulations on the passed manipulator. Add more manipulator options to this class as we make more manipulators faster.
    """

    def __init__(self, choices, manipulator):
        self.choices = {method: getattr(manipulator, method)
                        for method in choices if hasattr(manipulator, method)}
        self.choices_keys = list(self.choices.keys())

    def mutate(self, local_random, GB, manipulator):
        choice_key = local_random.choice(self.choices_keys)
        match choice_key:
            case "insert_atoms":
                new_system = manipulator.insert_atoms(
                    method="grid", num_to_insert=1)

            case "remove_atoms":
                new_system = manipulator.remove_atoms(num_to_remove=1)

            case "translate_right_grain":
                dz = (GB.z_dim / GB.repeat_factor[1]
                      ) * local_random.uniform(0, 1)
                dy = (GB.z_dim / GB.repeat_factor[0]
                      ) * local_random.uniform(0, 1)
                new_system = manipulator.translate_right_grain(dy=dy, dz=dz)
        return new_system


class MonteCarloMinimizer:
    """
    Minimizer class for finding the lowest energy configuration of a grain boundary.
    Runs a Monte-Carlo minimization approach on the provided GBMaker object, applying the provided manipulator options stochastically.
    """

    def __init__(self, GB: GBMaker, gb_energy_func: callable, choices: list, seed=time()):
        self.GB = GB
        self.gb_energy_func = gb_energy_func
        self.manipulator = GBManipulator(self.GB)
        self.mutator = Mutator(choices, self.manipulator)
        self.accepted_idx = [0]  # Initial guess is accepted by definition
        self.__operation_list__ = ["START"]
        self.local_random = np.random.default_rng(seed)
        self.manipulator.set_rng(self.local_random)
        self.GBE_vals = []

    def run_MC(self, E_accept=1e-1, max_steps=50, E_tol=1e-4, max_rejections=20, cooldown_rate=1.0, unique_id=uuid.uuid4()):
        # TODO: Add options for changing from linear to logarithmic cooldown
        """
        Runs an MC loop on the grain boundary structure till the set convergence criteria are met.
        The convergence creteria parameters are optional.
        :param max_steps: Integer value, sets the maximum number of iterations of MC that are run.
        :param E_accept: Energy increase value that should have a 50% chance of being accepted during the MC iterations (default value is in J/m^2).
        :param E_tol: Grain boundary energy decrease cut-off for terminating MC iterations (default value is in J/m^2).
        :return: Minimized energy value.
        """

        assert cooldown_rate > 0.0 and cooldown_rate <= 1.0

        # Get initial energy
        init_gbe, _ = self.gb_energy_func(
            self.GB,
            self.manipulator,
            self.manipulator.parents[0].whole_system,
            "initial"+str(unique_id),
        )
        self.GBE_vals += [init_gbe]

        T = -1 * E_accept / math.log(0.5)
        rejection_count = 0

        min_gbe = min(self.GBE_vals)

        for i in range(1, max_steps + 1):
            prev_gbe = self.GBE_vals[-1]

            new_system = self.mutator.mutate(
                self.local_random, self.GB, self.manipulator)

            new_gbe, dump_file_name = self.gb_energy_func(
                self.GB,
                self.manipulator,
                new_system,
                str(unique_id),
            )

            self.GBE_vals.append(new_gbe)

            accepted = new_gbe <= prev_gbe or self.local_random.uniform(
                0, 1) <= math.exp(-(new_gbe - prev_gbe) / T)

            if accepted:
                self.manipulator = GBManipulator(
                    dump_file_name,
                    unit_cell=self.GB.unit_cell,
                    gb_thickness=self.GB.gb_thickness,
                )
                self.manipulator.set_rng(self.local_random)
                prev_gbe = new_gbe

                if new_gbe <= min_gbe:
                    del_E = abs(min_gbe - new_gbe)
                    self.accepted_idx.append(i)
                    T *= cooldown_rate
                    rejection_count = 0
                    min_gbe = new_gbe
                    shutil.copyfile(dump_file_name, "min" + dump_file_name)
                    if del_E <= E_tol:
                        print("Meets energy tolerance criterion!")
                        break
                else:
                    rejection_count += 1
                    if rejection_count > max_rejections:
                        print("Too many rejections!")
                        break

        return min_gbe
