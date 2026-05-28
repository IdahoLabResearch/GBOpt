# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import inspect
import math
import shutil
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Any, Optional

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt.Checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    ENERGY_PENALTY,
    CandidateCheckpoint,
    CheckpointError,
    CheckpointStore,
    _wrap_batch_func_with_checkpoint,
)


class GBMinimizerError(Exception):
    """Base exception for the GBMinimizer module."""
    pass


class GBMinimizerTypeError(GBMinimizerError, TypeError):
    """Raised when an argument has an unexpected type."""
    pass


class GBMinimizerValueError(GBMinimizerError, ValueError):
    """Raised when an argument has an invalid value."""
    pass


class Mutator:
    """
    Mutator class for performing random manipulations on the passed manipulator.
    :param choices: A list of strings corresponding to GBManipulator operations.
    :param manipulator: A GBManipulator instance for mapping the choices list to GBmethod calls.
    """
    # TODO: Add more manipulator options to this class as we make more manipulators faster.

    def __init__(self, choices: list, manipulator: GBManipulator):
        self.choices = {method: getattr(manipulator, method)
                        for method in choices if hasattr(manipulator, method)}
        self.choices_keys = list(self.choices.keys())

    def mutate(self, local_random: np.random.default_rng, GB: GBMaker, manipulator: GBManipulator):
        """Performs a random mutation from the choices.
        :param local_random: A numpy.random.default_rng object for generating the random choices.
        "param GB: GBMaker object to get GB parameters for the mutation.
        :param GBManipulator: GBManipulator object to perform the mutation on.
        :return: Atom positions after the mutation."""
        choice_key = local_random.choice(self.choices_keys)
        mutation = None
        new_system = None
        match choice_key:
            case "insert_atoms":
                new_system = manipulator.insert_atoms(
                    method="grid", num_to_insert=1)
                mutation = "add1"

            case "remove_atoms":
                new_system = manipulator.remove_atoms(num_to_remove=1)
                mutation = "remove1"

            case "translate_right_grain":
                dz = (GB.z_dim / GB.repeat_factor[1]
                      ) * local_random.uniform(0, 1)
                dy = (GB.z_dim / GB.repeat_factor[0]
                      ) * local_random.uniform(0, 1)
                new_system = manipulator.translate_right_grain(dy=dy, dz=dz)
                mutation = f"shift{dy:.8f}dy{dz:.8f}dz"
            case _:
                raise ValueError(f"Unhandled mutation choice: {choice_key!r}")
        return mutation, new_system


class MonteCarloMinimizer:
    """
    Minimizer class for finding the lowest energy configuration of a grain boundary.
    Runs a Monte-Carlo minimization approach on the provided GBMaker object, applying the provided manipulator options stochastically.
    :param GB: GBMaker object to perform minimization on.
    :param gb_energy_func: A function that returns the energy of test GB structure. Currently expects a function
    that can be called with the params (GBMaker,GBManipulator,atom_positions,unique_id) .
    :param choices: A list of strings corresponding to GBManipulator operations. Used in setting up the Mutator class.
    :param seed: The seed to initialize the numpy.random.default_rng with.
    """

    def __init__(self, GB: GBMaker, gb_energy_func: Callable, choices: list, seed=None, *, initial_structure: Any = None):
        self.GB = GB
        self.gb_energy_func = gb_energy_func
        self.initial_structure = initial_structure
        self.manipulator = self._make_initial_manipulator()
        self.mutator = Mutator(choices, self.manipulator)
        self.accepted_idx = [0]  # Initial guess is accepted by definition
        self.operation_list = [["START", True]]
        self.local_random = np.random.default_rng(
            int(time()) if seed is None else seed)
        self.manipulator.rng = self.local_random
        self.GBE_vals = []

    def _make_initial_manipulator(self) -> GBManipulator:
        """
        Build the starting GBManipulator.

        - gbmaker (self.GB) remains the authoritative reference for unit_cell/gb_thickness.
        - initial structure may be:
            * None -> Use GBManipulator(self.GB)
            * GBMaker -> generate starting structure from that maker
            * anything else -> pass to GBManipulator as a "structure spec" that it can read,
                while still injecting unit_cell/gb_thickness from self.GB.
        """
        seed = self.initial_structure
        if seed is None:
            manip = GBManipulator(self.GB)
        elif isinstance(seed, GBMaker):
            manip = GBManipulator(seed)
        else:
            manip = GBManipulator(seed, unit_cell=self.GB.unit_cell,
                                  gb_thickness=self.GB.gb_thickness)

        return manip

    def run_MC(
        self,
        E_accept: float = 1e-1,
        min_steps: int = None,
        max_steps: int = 50,
        E_tol: float = 1e-4,
        max_rejections: int = 20,
        cooldown_rate: float = 1.0,
        unique_id: "int | uuid.UUID | None" = None,
        *,
        checkpoint_file: "str | Path | None" = None,
        checkpoint_format: str = "json",
        checkpoint_interval: int = 1,
        **kwargs
    ) -> float:
        # TODO: Add options for changing from linear to logarithmic cooldown
        """
        Runs an MC loop on the grain boundary structure till the set convergence
        criteria are met. The convergence criteria parameters are optional.
        :param E_accept: Energy increase value that should have a 50% chance of being
            accepted during the MC iterations (default value is in J/m^2).
        :param min_steps: Sets the minimum number of iterations of MC that are run.
            Defaults to None
        :param max_steps: Sets the maximum number of iterations of MC that are run.
        :param E_tol: Grain boundary energy decrease cut-off for terminating MC
            iterations (default value is in J/m^2).
        :param max_rejections: Maximum number of consequtive rejections before the MC
            iterations are terminated.
        :param cooldown_rate: Factor ((0,1]) by which to reduce the 'temperature' of
            the MC simulation each iteration.
        :param unique_id: Label for output files. Generated automatically if None;
            restored from checkpoint on resume.
        :param checkpoint_file: Path to checkpoint file. If the file exists, the run
            resumes saved structure, RNG state, temperature, accepted history,
            unique_id, min_steps, and cooldown_rate. On resume, max_steps may be
            increased to extend the run, and E_tol/max_rejections are applied from the
            current call. E_accept is only used for fresh runs because resumed runs
            restore T. run_params reflects the latest resume call for adjustable
            controls.
        :param checkpoint_format: Serialization format for the checkpoint file. Either
            ``"json"`` (default, human-readable) or ``"pickle"`` (binary, no numpy
            conversion needed).
        :param checkpoint_interval: Save a checkpoint every N steps (default 1, i.e.
            every step).
        :param **kwargs: Keyword arguments that are passed to gb_energy_func
        :return: Minimized energy value.
        """

        assert cooldown_rate > 0.0 and cooldown_rate <= 1.0

        try:
            checkpoint = CheckpointStore.from_optional(
                checkpoint_file, checkpoint_format, checkpoint_interval
            )
        except CheckpointError as e:
            raise GBMinimizerValueError(str(e)) from e

        type_dict = {value: key for key,
                     value in self.GB.unit_cell.type_map.items()}

        try:
            state = checkpoint.load()
        except CheckpointError as e:
            raise GBMinimizerError(str(e)) from e

        if state is not None:
            self.GBE_vals = state["state"]["GBE_vals"]
            self.accepted_idx = state["state"]["accepted_idx"]
            self.operation_list = state["state"]["operation_list"]
            self.local_random.bit_generator.state = state["rng_state"]
            unique_id = state["run_params"]["unique_id"]
            min_steps = state["run_params"]["min_steps"]
            cooldown_rate = state["run_params"]["cooldown_rate"]
            _resume_step = state["progress_index"] + 1
            T = state["state"]["T"]
            rejection_count = state["state"]["rejection_count"]
            min_gbe = state["best_energy"]
            prev_gbe = state["state"]["prev_gbe"]
            best_dump = state["best_dump"]
            _current_dump = state["state"]["current_structure_dump"]
            self.manipulator = GBManipulator(
                _current_dump,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
                type_dict=type_dict,
            )
            self.manipulator.rng = self.local_random
        else:
            _resume_step = 1
            unique_id = str(uuid.uuid4()) if unique_id is None else str(unique_id)
            init_system = np.array(
                self.manipulator.parents[0].whole_system, copy=True)
            init_gbe, _current_dump = self.gb_energy_func(
                self.GB,
                self.manipulator,
                init_system,
                "initial" + str(unique_id),
                **kwargs,
            )
            self.GBE_vals.append(init_gbe)
            T = -1 * E_accept / math.log(0.5)
            rejection_count = 0
            min_gbe = min(self.GBE_vals)
            prev_gbe = init_gbe
            best_dump = None

        def _build_state(step):
            # Note that E_tol, max_rejections, and E_accept can be changed on resume;
            # run_params reflects the latest resume call for adjustable controls.
            return {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "minimizer": "MonteCarloMinimizer",
                "progress_unit": "step",
                "progress_index": step,
                "best_energy": min_gbe,
                "best_dump": str(best_dump) if best_dump else None,
                "rng_state": self.local_random.bit_generator.state,
                "run_params": {
                    "E_accept": E_accept,
                    "min_steps": min_steps,
                    "max_steps": max_steps,
                    "E_tol": E_tol,
                    "max_rejections": max_rejections,
                    "cooldown_rate": cooldown_rate,
                    "unique_id": str(unique_id),
                },
                "state": {
                    "T": T,
                    "rejection_count": rejection_count,
                    "prev_gbe": prev_gbe,
                    "current_structure_dump": str(_current_dump),
                    "GBE_vals": self.GBE_vals,
                    "accepted_idx": self.accepted_idx,
                    "operation_list": self.operation_list,
                },
            }

        _last_completed_step = state["progress_index"] if state is not None else -1
        _early_exit = False
        for i in range(_resume_step, max_steps + 1):
            mutation, new_system = self.mutator.mutate(
                self.local_random, self.GB, self.manipulator)

            new_gbe, dump_file_name = self.gb_energy_func(
                self.GB,
                self.manipulator,
                new_system,
                str(unique_id),
                **kwargs,
            )
            self.GBE_vals.append(new_gbe)

            accepted = new_gbe <= prev_gbe or self.local_random.uniform(
                0, 1) <= math.exp(-(new_gbe - prev_gbe) / T)

            if accepted:
                self.operation_list.append([mutation, True])
                self.manipulator = GBManipulator(
                    dump_file_name,
                    unit_cell=self.GB.unit_cell,
                    gb_thickness=self.GB.gb_thickness,
                    type_dict=type_dict,
                )
                self.manipulator.rng = self.local_random
                _current_dump = dump_file_name
                prev_gbe = new_gbe
                self.accepted_idx.append(i)
                rejection_count = 0

                if new_gbe <= min_gbe:
                    best_dump = Path(dump_file_name).with_name(
                        "min_" + Path(dump_file_name).name)
                    shutil.copyfile(dump_file_name, best_dump)
                    del_E = min_gbe - new_gbe
                    min_gbe = new_gbe
                    if 0 < del_E <= E_tol and (min_steps is None or i >= min_steps):
                        print("Meets energy tolerance criterion")
                        checkpoint.save_final(_build_state(i))
                        _early_exit = True
                        break
            else:
                self.operation_list.append([mutation, False])
                rejection_count += 1
                if rejection_count > max_rejections:
                    print("Too many rejections!")
                    T *= cooldown_rate
                    checkpoint.save_final(_build_state(i))
                    _early_exit = True
                    break

            T *= cooldown_rate

            _last_completed_step = i
            checkpoint.save_if_due(i, lambda: _build_state(i))
        if not _early_exit and _last_completed_step >= 0:
            checkpoint.save_final(_build_state(_last_completed_step))

        return min_gbe


class GeneticAlgorithmMinimizer:
    """
    Minimizer class for finding the lowest energy configuration of a grain boundary
    using a simple genetic algorithm(GA).
    Mirrors the interface of MonteCarloMinimizer while using GA operations to explore
    the configuration space.
    """

    def __init__(self, GB: GBMaker, gb_energy_func: Callable, choices: list, seed=None, *, initial_structure: Any = None, population_size: int = 20, generations: int = 50, keep_top_pct: int = 10, intermediate_pct: int = 60, gb_batch_energy_func: Callable | None = None):
        """
        : param GB: GBMaker object to perform minimization on.
        : param gb_energy_func: Function that returns the energy of a GB structure. It
            must be callable with (GBMaker, GBManipulator, atom_positions, unique_id).
        : param choices: List of strings corresponding to GBManipulator operations. Used
            to configure the Mutator.
        : param seed: Seed for numpy.random.default_rng. Keyword argument, optional,
            defaults to the current time.
        : param population_size: Number of candidates per generation. Keyword argument,
            optional, defaults to 20.
        : param generations: Number of generations to iterate. Keyword argument,
            optional, defaults to 50.
        : param keep_top_pct: Percentage of lowest-energy structures carried over
            unchanged. Keyword argument, optional, defaults to 10.
        : param intermediate_pct: Percentage of structures eligible for
            crossover/mutation selection. Keyword argument, optional, defaults to 60.
        : param gb_batch_energy_func: Optional batch-evaluation function for processing a
            population in one call. It should accept(GBMaker, manipulators,
            atom_positions_list, lineages, unique_ids) and return a list of dictionaries
            containing at least ``"energy"`` and ``"final_dump"`` keys. If not provided,
            fall back to calling ``gb_energy_func`` per candidate. If the function does
            not declare a ``checkpoint`` keyword argument it is automatically wrapped so
            that checkpointing still occurs at batch-return granularity; a
            : class: `~warnings.UserWarning` is emitted in that case. Declare
            ``checkpoint = None`` and call ``checkpoint.record(unique_id, energy, dump)``
            per job to get per-job recovery granularity.
        """
        self.GB = GB
        self.gb_energy_func = gb_energy_func
        if gb_batch_energy_func is not None:
            try:
                sig = inspect.signature(gb_batch_energy_func)
                if "checkpoint" not in sig.parameters:
                    warnings.warn(
                        "gb_batch_energy_func does not accept a 'checkpoint' kwarg. "
                        "It has been automatically wrapped so checkpointing occurs at "
                        "batch-return granularity. For per-job recovery, add "
                        "'checkpoint=None' to your batch function signature and call "
                        "checkpoint.record(unique_id, energy, dump) as each job completes.",
                        UserWarning,
                        stacklevel=2,
                    )
                    gb_batch_energy_func = _wrap_batch_func_with_checkpoint(
                        gb_batch_energy_func)
            except ValueError:
                # C callables have no inspectable signature — wrap at batch-return granularity.
                warnings.warn(
                    "gb_batch_energy_func signature could not be inspected. "
                    "It has been automatically wrapped so checkpointing occurs at "
                    "batch-return granularity.",
                    UserWarning,
                    stacklevel=2,
                )
                gb_batch_energy_func = _wrap_batch_func_with_checkpoint(
                    gb_batch_energy_func)
            except TypeError:
                raise GBMinimizerTypeError(
                    "gb_batch_energy_func must be callable."
                )
        self.gb_batch_energy_func = gb_batch_energy_func
        self.history = []
        self.initial_structure = initial_structure
        self.local_random = np.random.default_rng(
            int(time()) if seed is None else seed
        )
        self.manipulator = self._make_initial_manipulator()
        self.mutator = Mutator(choices, self.manipulator)
        self.manipulator.rng = self.local_random
        self.population_size = population_size
        self.generations = generations
        self.keep_top_pct = keep_top_pct
        self.intermediate_pct = intermediate_pct
        self.GBE_vals = []

    def _make_initial_manipulator(self) -> GBManipulator:
        seed = self.initial_structure
        if seed is None:
            manip = GBManipulator(self.GB)
        elif isinstance(seed, GBMaker):
            manip = GBManipulator(seed)
        else:
            manip = GBManipulator(seed, unit_cell=self.GB.unit_cell,
                                  gb_thickness=self.GB.gb_thickness)

        manip.rng = self.local_random

        return manip

    def _make_manipulator_from_file(self, filename: str) -> GBManipulator:
        manipulator = GBManipulator(
            filename,
            unit_cell=self.GB.unit_cell,
            gb_thickness=self.GB.gb_thickness,
        )
        manipulator.rng = self.local_random
        return manipulator

    def _select_indices_by_energy(self, energies: list) -> tuple[list[int], list[int]]:
        idx_sorted = sorted(range(len(energies)), key=lambda i: energies[i])

        n_top = max(0, (len(energies) * self.keep_top_pct) // 100)
        n_inter = max(0, (len(energies) * self.intermediate_pct) // 100)

        lowest_top = idx_sorted[:n_top]
        intermediate = idx_sorted[:n_inter]
        return lowest_top, intermediate

    def _evaluate_generation(
        self,
        population_manipulators: list[GBManipulator],
        population_structures: list[np.ndarray],
        population_lineages: list[list[str]],
        gen: int,
        unique_id: int,
        gen_checkpoint: Optional[CandidateCheckpoint] = None,
    ) -> tuple[list[float], list[Optional[str]], list[Optional[GBManipulator]]]:
        """Evaluate all candidates, optionally using a batch energy function.

        : param gen_checkpoint: If provided, already-evaluated candidates are skipped
            and new results are recorded after each evaluation.
        """

        all_uids = [
            f"GA_{unique_id}_g{gen}_c{i}"
            for i in range(len(population_structures))
        ]

        if self.gb_batch_energy_func is not None:
            if gen_checkpoint is not None:
                pending = [
                    (i, u) for i, u in enumerate(all_uids)
                    if not gen_checkpoint.is_done(u)
                ]
                if pending:
                    pending_idxs, pending_uids = zip(*pending)
                    pending_idxs = list(pending_idxs)
                    pending_uids = list(pending_uids)
                    new_results = self.gb_batch_energy_func(
                        self.GB,
                        [population_manipulators[i] for i in pending_idxs],
                        [population_structures[i] for i in pending_idxs],
                        [population_lineages[i] for i in pending_idxs],
                        pending_uids,
                        checkpoint=gen_checkpoint,
                    )
                    # Record any results the batch func did not record itself
                    for uid, result in zip(pending_uids, new_results):
                        if not gen_checkpoint.is_done(uid):
                            gen_checkpoint.record(
                                uid,
                                float(result.get("energy", ENERGY_PENALTY)),
                                result.get("final_dump", None),
                            )
                batch_results = [
                    {
                        "energy": gen_checkpoint.get_result(u)[0],
                        "final_dump": gen_checkpoint.get_result(u)[1],
                    }
                    for u in all_uids
                ]
            else:
                batch_results = self.gb_batch_energy_func(
                    self.GB,
                    population_manipulators,
                    population_structures,
                    population_lineages,
                    all_uids,
                )

            gen_energies = []
            gen_files = []
            evaluated_manipulators = []
            for result in batch_results:
                energy = float(result.get("energy", ENERGY_PENALTY))
                dump = result.get("final_dump", None)

                gen_energies.append(energy)
                if self._is_valid_file(dump):
                    gen_files.append(dump)
                    try:
                        evaluated_manipulators.append(
                            self._make_manipulator_from_file(dump))
                    except Exception:
                        gen_files[-1] = None
                        gen_energies[-1] = ENERGY_PENALTY
                        evaluated_manipulators.append(None)
                else:
                    gen_files.append(None)
                    gen_energies[-1] = ENERGY_PENALTY
                    evaluated_manipulators.append(None)

            return gen_energies, gen_files, evaluated_manipulators

        gen_energies: list[float] = []
        gen_files: list[Optional[str]] = []
        evaluated_manipulators: list[Optional[GBManipulator]] = []

        for idx, (manipulator, atom_positions) in enumerate(
                zip(population_manipulators, population_structures)):
            uid = all_uids[idx]
            if gen_checkpoint is not None and gen_checkpoint.is_done(uid):
                gbe, dump_file_name = gen_checkpoint.get_result(uid)
            else:
                try:
                    gbe, dump_file_name = self.gb_energy_func(
                        self.GB, manipulator, atom_positions, uid)
                except Exception:
                    gbe, dump_file_name = ENERGY_PENALTY, None
                if gen_checkpoint is not None:
                    gen_checkpoint.record(uid, gbe, dump_file_name)

            gen_energies.append(float(gbe))
            if self._is_valid_file(dump_file_name):
                gen_files.append(dump_file_name)
                try:
                    evaluated_manipulators.append(
                        self._make_manipulator_from_file(dump_file_name))
                except Exception:
                    gen_files[-1] = None
                    gen_energies[-1] = ENERGY_PENALTY
                    evaluated_manipulators.append(None)
            else:
                gen_files.append(None)
                gen_energies[-1] = ENERGY_PENALTY
                evaluated_manipulators.append(None)

        return gen_energies, gen_files, evaluated_manipulators

    def _make_next_generation(self, files: list[str], intermediate_indices: list[int]) -> tuple[list[GBManipulator], list[np.ndarray], list[list[str]]]:
        if not files:
            raise ValueError(
                "No valid parent files provided to _make_next_generation().")

        if not intermediate_indices:
            intermediate_indices = list(range(len(files)))
        candidates: list[np.ndarray] = []
        manipulators: list[GBManipulator] = []
        lineages: list[list[str]] = []

        N_slice = self.population_size // 2
        N_mutate = self.population_size - N_slice

        # Slice & merge
        for _ in range(N_slice):
            replace = len(intermediate_indices) < 2
            idx_1, idx_2 = self.local_random.choice(
                intermediate_indices, size=2, replace=replace)
            p1, p2 = files[idx_1], files[idx_2]
            new_manip = GBManipulator(
                p1,
                p2,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
            )
            new_manip.rng = self.local_random
            new_struct = new_manip.slice_and_merge()

            candidates.append(new_struct)
            manipulators.append(new_manip)
            lineages.append(["slice_and_merge", p1, p2])

        # Mutations
        if not intermediate_indices:
            intermediate_indices = list(range(len(files)))
        choices = self.local_random.choice(
            intermediate_indices, size=N_mutate, replace=True)
        for idx in choices:
            parent = files[idx]
            new_manip = GBManipulator(
                parent,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
            )
            new_manip.rng = self.local_random
            mutation, new_struct = self.mutator.mutate(
                local_random=self.local_random,
                GB=self.GB,
                manipulator=new_manip,
            )

            candidates.append(new_struct)
            manipulators.append(new_manip)
            lineages.append([mutation, parent])

        return manipulators, candidates, lineages

    def _is_valid_file(self, p: Optional[str]) -> bool:
        return bool(p) and Path(p).is_file()

    def run_GA(
        self,
        unique_id: "int | uuid.UUID | None" = None,
        *,
        checkpoint_file: "str | Path | None" = None,
        checkpoint_format: str = "json",
        checkpoint_interval: int = 1
    ) -> tuple:
        """
        Runs a genetic algorithm loop on the grain boundary structure.

        Checkpointing is optional. Pass * checkpoint_file * to enable it; omit it (or pass
        ``None``) to to run without any checkpoint file. When enabled, a per-candidate
        sidecar(``{stem}.iter{N}{ext}``) is also written so a mid-generation crash can
        be resumed without re-evaluating completed candidates. The checkpoint file is
        **not ** deleted on normal completion — it can be used to continue the run later
        by calling ``run_GA`` again with the same * checkpoint_file * after increasing
        *generations*. The checkpoint file and the sibling ``*.pending`` structure files
        in the same directory form a unit - both must be present to resume or extend a
        run. Do not delete or move the ``.pending`` files independently of the
        checkpoint file.

        : param unique_id: Label applied to all output files. Restored from the
            checkpoint on resume if not provided.
        : param checkpoint_file: Path to the run-level checkpoint file.  If the file
            exists the run resumes from it; otherwise a fresh run begins and the file
            is created.
        : param checkpoint_format: Serialization format — ``"json"`` (default,
            human-readable) or ``"pickle"`` (binary, no numpy conversion needed).
        : param checkpoint_interval: Save a run-level checkpoint every N generations
            (default 1).
        : return: Tuple containing the minimum energy value observed and the
            associated dump filename.
        """

        try:
            if checkpoint_file is not None:
                checkpoint_file = Path(checkpoint_file)
                checkpoint = CheckpointStore.from_optional(
                    checkpoint_file, checkpoint_format, checkpoint_interval
                )
                try:
                    state = checkpoint.load()
                except CheckpointError as e:
                    raise GBMinimizerError(str(e)) from e
                if state is not None:
                    unique_id = state["run_params"]["unique_id"]
                else:
                    unique_id = str(unique_id) if unique_id is not None else str(
                        uuid.uuid4())
            else:
                unique_id = str(unique_id) if unique_id is not None else str(
                    uuid.uuid4())
                checkpoint = CheckpointStore.disabled()
                state = None
        except CheckpointError as e:
            raise GBMinimizerValueError(str(e)) from e

        if state is not None:
            self.GBE_vals = state["state"]["GBE_vals"]
            self.history = state["state"]["history"]
            self.local_random.bit_generator.state = state["rng_state"]
            _start_gen = state["progress_index"] + 1
            best_energy = state["best_energy"]
            best_dump = state["best_dump"]
            # Drop any stale iter checkpoint for the just-completed generation
            stale = CandidateCheckpoint._derive_path(
                checkpoint_file, state["progress_index"])
            if stale.exists():
                stale.unlink()
            population_lineages = state["state"]["population_lineages"]
            population_checkpoint_paths = state["state"].get(
                "population_checkpoint_paths",
                [lin[1] for lin in state["state"]["population_lineages"]]
            )
            population_manipulators = []
            population_structures = []
            for cp_path in population_checkpoint_paths:
                try:
                    manip = self._make_manipulator_from_file(cp_path)
                except Exception:
                    raise GBMinimizerError(
                        f"Checkpoint population path {cp_path} is missing/unreadable.")
                population_manipulators.append(manip)
                population_structures.append(
                    np.array(manip.parents[0].whole_system, copy=True)
                )
        else:
            # Evaluate the initial structure
            init_system = np.array(
                self.manipulator.parents[0].whole_system, copy=True)
            init_gbe, init_dump = self.gb_energy_func(
                self.GB,
                self.manipulator,
                init_system,
                "GA_initial" + str(unique_id),
            )
            self.GBE_vals.append([init_gbe])
            self.history = []

            best_energy = init_gbe
            best_dump = init_dump

            base_parent = init_dump
            population_manipulators = []
            population_structures = []
            population_lineages = []

            if self.initial_structure is not None:
                seed_manip = self._make_manipulator_from_file(base_parent)
                population_manipulators.append(seed_manip)
                population_structures.append(
                    np.array(seed_manip.parents[0].whole_system, copy=True))
                population_lineages.append(["START", base_parent])

            n_to_generate = self.population_size - len(population_manipulators)
            for _ in range(n_to_generate):
                candidate_manip = self._make_manipulator_from_file(base_parent)
                mutation, candidate_struct = self.mutator.mutate(
                    local_random=self.local_random,
                    GB=self.GB,
                    manipulator=candidate_manip,
                )
                population_manipulators.append(candidate_manip)
                population_structures.append(candidate_struct)
                population_lineages.append([mutation, base_parent])

            population_checkpoint_paths = [lin[1] for lin in population_lineages]

            _start_gen = 0

        def _build_ga_state(gen):
            return {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "minimizer": "GeneticAlgorithmMinimizer",
                "progress_unit": "generation",
                "progress_index": gen,
                "best_energy": best_energy,
                "best_dump": best_dump,
                "rng_state": self.local_random.bit_generator.state,
                "run_params": {"unique_id": str(unique_id)},
                "state": {
                    "GBE_vals": self.GBE_vals,
                    "history": self.history,
                    "population_lineages": population_lineages,
                    "population_checkpoint_paths": population_checkpoint_paths
                },
            }

        _current_pending = []
        _last_completed_gen = -1
        # Main GA loop
        for gen in range(_start_gen, self.generations):
            if checkpoint.enabled:
                _current_pending = [
                    p for p in population_checkpoint_paths
                    if str(p).endswith(".pending")
                ]
            all_uids = [
                f"GA_{unique_id}_g{gen}_c{i}"
                for i in range(len(population_manipulators))
            ]
            gen_checkpoint = (
                CandidateCheckpoint.new_or_resume(
                    checkpoint_file, checkpoint_format, gen, all_uids)
                if checkpoint.enabled else None
            )

            gen_energies, gen_files, evaluated_manipulators = self._evaluate_generation(
                population_manipulators,
                population_structures,
                population_lineages,
                gen,
                unique_id,
                gen_checkpoint=gen_checkpoint,
            )

            valid_old_idxs = [i for i, f in enumerate(
                gen_files) if self._is_valid_file(f)]

            self.GBE_vals.append(gen_energies)
            self.history.append(list(zip(population_lineages, gen_energies)))

            if not valid_old_idxs:
                # If nothing valid survived evaluation, re-seed from best
                next_manipulators = []
                next_structures = []
                next_lineages = []

                for _ in range(self.population_size):
                    candidate_manip = self._make_manipulator_from_file(
                        best_dump)
                    mutation, candidate_struct = self.mutator.mutate(
                        local_random=self.local_random, GB=self.GB, manipulator=candidate_manip)
                    next_manipulators.append(candidate_manip)
                    next_structures.append(candidate_struct)
                    next_lineages.append([mutation, best_dump])

                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
            else:
                for i in valid_old_idxs:
                    gbe = gen_energies[i]
                    dump_file_name = gen_files[i]
                    if gbe < best_energy:
                        best_energy = gbe
                        best_dump = dump_file_name

                # Build compressed arrays of only valid candidates for selection and breeding
                valid_energies = [gen_energies[i] for i in valid_old_idxs]
                valid_files = [gen_files[i] for i in valid_old_idxs]

                # Selection
                lowest_valid_idxs, inter_valid_idxs = self._select_indices_by_energy(
                    valid_energies)

                # Carry over lowest energies
                next_manipulators = []
                next_structures = []
                next_lineages = []
                for j in lowest_valid_idxs:
                    old_idx = valid_old_idxs[j]
                    manip = evaluated_manipulators[old_idx]
                    dump = gen_files[old_idx]
                    if manip is None or dump is None:
                        continue
                    next_manipulators.append(manip)
                    next_structures.append(manip.parents[0].whole_system)
                    next_lineages.append(["carryover", dump])

                valid_files_str = [f for f in valid_files if f is not None]

                new_manips, new_structs, new_lineages = self._make_next_generation(
                    valid_files_str,
                    inter_valid_idxs,
                )

                next_manipulators.extend(new_manips)
                next_structures.extend(new_structs)
                next_lineages.extend(new_lineages)

                population_manipulators = next_manipulators[:self.population_size]
                population_structures = next_structures[:self.population_size]
                population_lineages = next_lineages[:self.population_size]

            _last_completed_gen = gen
            is_final_gen = (gen == self.generations - 1)
            if checkpoint.enabled and (checkpoint.is_due(gen + 1) or is_final_gen):
                new_pending = []
                for i, (manip, struct) in enumerate(
                    zip(population_manipulators, population_structures)
                ):
                    pending_path = str(
                        checkpoint_file.parent
                        / f"GA_{unique_id}_g{gen + 1}_c{i}.pending"
                    )
                    self.GB.write_lammps(
                        pending_path, struct, manip.parents[0].box_dims
                    )
                    new_pending.append(pending_path)
                population_checkpoint_paths = new_pending
                checkpoint.save_final(_build_ga_state(gen))
                for p in _current_pending:
                    Path(p).unlink(missing_ok=True)
                _current_pending = new_pending

            # Iter checkpoint is transient; main checkpoint covers this boundary
            if gen_checkpoint is not None:
                gen_checkpoint.delete()

        return (best_energy, best_dump)
