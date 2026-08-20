# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import copy as copy_module
import inspect
import math
import shutil
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt._candidate_admissibility import (
    CandidateAdmissibilityError,
    validate_formula_composition,
)
from GBOpt._explicit_ownership_evaluation import (
    CandidateEvaluation,
    ExplicitOwnershipEvaluator,
)
from GBOpt.Checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CandidateCheckpoint,
    CheckpointError,
    CheckpointStore,
    _wrap_batch_func_with_checkpoint,
)
from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnership,
    GrainOwnershipError,
    LammpsDataError,
)
from GBOpt.GBMaker import GBMakerError
from GBOpt.GBManipulator import (
    CompositionAwareCrossoverError,
    GBManipulatorError,
    GBManipulatorValueError,
    ParentError,
)

ENERGY_PENALTY: float = 1.0e30
"""Optimizer policy for ranking failed candidate evaluations."""

_OWNED_GA_CHECKPOINT_VERSION = 3


@dataclass(frozen=True, slots=True)
class _CachedEvaluation:
    """Reusable result for one unchanged legacy-path carryover candidate."""

    energy: float
    structure_path: str


def _candidate_mapping_to_state(mapping: CandidateFileMapping) -> dict:
    """Serialize a candidate-local ownership mapping for checkpoint persistence.

    :param mapping: Validated candidate/file mapping.
    :return: JSON-safe mapping state without live optimizer objects.
    """
    return {
        "atom_ids": mapping.atom_ids,
        "labels": mapping.labels,
        "species": mapping.species.tolist(),
        "box_dims": mapping.box_dims,
        "gb_plane_x": mapping.gb_plane_x,
        "inplane_periodic": mapping.inplane_periodic,
        "left_grain_x_bounds": mapping.left_grain_x_bounds,
        "right_grain_x_bounds": mapping.right_grain_x_bounds,
        "coordinate_tolerance": mapping.coordinate_tolerance,
        "normal_topology": mapping.normal_topology.value,
    }


def _candidate_mapping_from_state(state: object) -> CandidateFileMapping:
    """Reconstruct and validate a checkpointed candidate/file mapping.

    :param state: Deserialized mapping state.
    :return: Validated candidate-local ownership mapping.
    :raises GrainOwnershipError: If the checkpointed mapping is malformed.
    """
    if not isinstance(state, dict):
        raise GrainOwnershipError("candidate mapping state must be a dictionary")
    try:
        return CandidateFileMapping(
            atom_ids=np.asarray(state["atom_ids"], dtype=object),
            labels=np.asarray(state["labels"], dtype=object),
            species=np.asarray(state["species"], dtype=object),
            box_dims=np.asarray(state["box_dims"], dtype=object),
            gb_plane_x=state["gb_plane_x"],
            inplane_periodic=tuple(state["inplane_periodic"]),
            left_grain_x_bounds=np.asarray(
                state["left_grain_x_bounds"], dtype=object
            ),
            right_grain_x_bounds=np.asarray(
                state["right_grain_x_bounds"], dtype=object
            ),
            coordinate_tolerance=state["coordinate_tolerance"],
            normal_topology=state["normal_topology"],
        )
    except (KeyError, TypeError) as exc:
        raise GrainOwnershipError(
            "candidate mapping checkpoint state is incomplete or malformed"
        ) from exc


class GBMinimizerError(Exception):
    """Base exception for the GBMinimizer module."""


class GBMinimizerTypeError(GBMinimizerError, TypeError):
    """Raised when an argument has an unexpected type."""


class GBMinimizerValueError(GBMinimizerError, ValueError):
    """Raised when an argument has an invalid value."""


class Mutator:
    """Perform randomly selected manipulations on a GB candidate.

    :param choices: Mutation operation names to make available.
    :param manipulator: GBManipulator used to validate the requested operations.
    """

    # TODO: Add more manipulator options to this class as we make more
    # manipulators faster.

    def __init__(self, choices: list[str], manipulator: GBManipulator):
        invalid_choices = [
            method for method in choices if not hasattr(manipulator, method)
        ]
        if invalid_choices:
            raise GBMinimizerValueError(
                "Unknown GBManipulator mutation choice(s): "
                + ", ".join(repr(choice) for choice in invalid_choices)
            )

        # Duplicate names do not weight a mutation more heavily.
        self.choices_keys = list(dict.fromkeys(choices))
        if not self.choices_keys:
            raise GBMinimizerValueError(
                "At least one mutation choice must be provided."
            )

    def _apply_mutation(
        self,
        choice_key: str,
        *,
        local_random: np.random.Generator,
        GB: GBMaker,
        manipulator: GBManipulator,
    ):
        """Apply one explicitly selected mutation.

        :param choice_key: Mutation operation to apply.
        :param local_random: Optimizer-owned random-number generator.
        :param GB: GBMaker providing boundary dimensions and repeat factors.
        :param manipulator: GBManipulator on which to perform the mutation.
        :return: Mutation description and resulting atom positions.
        :raises GBManipulatorValueError: If the selected mutation is infeasible.
        :raises GBMinimizerValueError: If ``choice_key`` is unsupported.
        """
        match choice_key:
            case "insert_atoms":
                new_system = manipulator.insert_atoms(
                    method="grid",
                    num_to_insert=1,
                )
                mutation = "add1"

            case "remove_atoms":
                new_system = manipulator.remove_atoms(num_to_remove=1)
                mutation = "remove1"

            case "translate_right_grain":
                parent = manipulator.parents[0]
                y_dim = parent.box_dims[1, 1] - parent.box_dims[1, 0]
                z_dim = parent.box_dims[2, 1] - parent.box_dims[2, 0]

                dy = (y_dim / GB.repeat_factor[0]) * local_random.uniform(0, 1)
                dz = (z_dim / GB.repeat_factor[1]) * local_random.uniform(0, 1)

                new_system = manipulator.translate_right_grain(dy=dy, dz=dz)
                mutation = f"shift{dy:.8f}dy{dz:.8f}dz"

            case _:
                raise GBMinimizerValueError(
                    f"Unhandled mutation choice: {choice_key!r}"
                )

        return mutation, new_system

    def mutate(
        self,
        local_random: np.random.Generator,
        GB: GBMaker,
        manipulator: GBManipulator,
    ):
        """Perform a randomly selected feasible mutation.

        Each configured mutation is attempted at most once. If an operation is
        physically infeasible for the current candidate, another configured operation is
        tried. Failure of every configured operation is fatal.

        :param local_random: Optimizer-owned random-number generator.
        :param GB: GBMaker providing boundary dimensions and repeat factors.
        :param manipulator: GBManipulator on which to perform the mutation.
        :return: Mutation description and resulting atom positions.
        :raises GBMinimizerError: If no configured mutation can produce a candidate.
        """
        choice_order = local_random.permutation(len(self.choices_keys))
        failures: list[tuple[str, GBManipulatorValueError]] = []

        for choice_index in choice_order:
            choice_key = self.choices_keys[int(choice_index)]
            try:
                return self._apply_mutation(
                    choice_key,
                    local_random=local_random,
                    GB=GB,
                    manipulator=manipulator,
                )
            except GBManipulatorValueError as exc:
                failures.append((choice_key, exc))

        failure_details = "; ".join(
            f"{choice}: {exc}" for choice, exc in failures
        )
        error = GBMinimizerError(
            "No configured mutation could produce a valid candidate. Attempted "
            f"mutations: {failure_details}"
        )
        raise error from failures[-1][1]


class MonteCarloMinimizer:
    """
    Minimizer class for finding the lowest energy configuration of a grain boundary.
    Runs a Monte-Carlo minimization approach on the provided GBMaker object, applying
    the provided manipulator options stochastically.
    :param GB: GBMaker object to perform minimization on.
    :param gb_energy_func: A function that returns the energy of test GB structure.
        Currently expects a function that can be called with the params
        (GBMaker,GBManipulator,atom_positions,unique_id) .
    :param choices: A list of strings corresponding to GBManipulator operations. Used in
        setting up the Mutator class.
    :param seed: The seed to initialize the numpy.random.default_rng with.
    """

    def __init__(
        self,
        GB: GBMaker,
        gb_energy_func: Callable,
        choices: list,
        seed=None,
        *,
        initial_structure: Any = None,
    ):
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

        - gbmaker (self.GB) remains the authoritative reference for
          unit_cell/gb_thickness.
        - initial structure may be:
          * None -> Use GBManipulator(self.GB)
          * GBMaker -> generate starting structure from that maker
          * anything else -> pass to GBManipulator as a "structure spec" that it can
            read, while still injecting unit_cell/gb_thickness from self.GB.
        """
        seed = self.initial_structure
        if seed is None:
            manip = GBManipulator(self.GB)
        elif isinstance(seed, GBMaker):
            manip = GBManipulator(seed)
        else:
            manip = GBManipulator(
                seed, unit_cell=self.GB.unit_cell, gb_thickness=self.GB.gb_thickness
            )

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
        **kwargs,
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
                self.local_random, self.GB, self.manipulator
            )

            new_gbe, dump_file_name = self.gb_energy_func(
                self.GB,
                self.manipulator,
                new_system,
                str(unique_id),
                **kwargs,
            )
            self.GBE_vals.append(new_gbe)

            accepted = new_gbe <= prev_gbe or self.local_random.uniform(
                0, 1
            ) <= math.exp(-(new_gbe - prev_gbe) / T)

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
    using a simple genetic algorithm (GA). Mirrors the interface of MonteCarloMinimizer
    while using GA operations to explore the configuration space.
    """

    def __init__(
        self,
        GB: GBMaker,
        gb_energy_func: Callable,
        choices: list,
        seed=None,
        *,
        initial_structure: GBMaker | str | Path | None = None,
        initial_ownership: GrainOwnership | None = None,
        allow_variable_cell: bool = False,
        population_size: int = 20,
        generations: int = 50,
        keep_top_pct: int = 10,
        intermediate_pct: int = 60,
        slice_and_merge_pct: float = 50.0,
        reuse_carryover_evaluations: bool = False,
        gb_batch_energy_func: Callable | None = None,
        crossover_surface: str = "periodic_wave",
        crossover_max_tilt_degrees: float = 5.0,
        crossover_attempts: int = 8,
    ):
        """
        :param GB: GBMaker object to perform minimization on.
        :param gb_energy_func: Function that returns the energy of a GB structure. It
            must be callable with (GBMaker, GBManipulator, atom_positions, unique_id).
        :param choices: List of strings corresponding to GBManipulator operations. Used
            to configure the Mutator.
        :param seed: Seed for numpy.random.default_rng. Keyword argument, optional,
            defaults to the current time.
        :param initial_structure: Keyword argument, optional, defaults to ``None``.
            GBMaker or file-backed initial structure.
        :param initial_ownership: Keyword argument, optional, defaults to ``None``.
            Explicit ownership aligned to atom IDs in a file-backed initial structure.
        :param allow_variable_cell: Keyword argument, optional, defaults to ``False``.
            Allow orthogonal box dimensions returned by explicit-ownership evaluators
            to evolve between GA generations. Requires ``initial_ownership``.
        :param population_size: Number of candidates per generation. Keyword argument,
            optional, defaults to 20.
        :param generations: Number of generations to iterate. Keyword argument,
            optional, defaults to 50.
        :param keep_top_pct: Percentage of lowest-energy structures carried over
            unchanged. Keyword argument, optional, defaults to 10.
        :param intermediate_pct: Percentage of structures eligible for
            crossover/mutation selection. Keyword argument, optional, defaults to 60.
        :param slice_and_merge_pct: Percentage of non-carryover offspring generated by
            slice-and-merge crossover. The remaining offspring are generated by
            mutation. Keyword argument, optional, defaults to 50.0.
        :param reuse_carryover_evaluations: Reuse the validated energy and relaxed
            artifact of unchanged successful carryover candidates instead of invoking
            the evaluator again. Keyword argument, optional, defaults to ``False``.
        :param gb_batch_energy_func: Optional batch-evaluation function for processing a
            population in one call. It should accept (GBMaker, manipulators,
            atom_positions_list, lineages, unique_ids) and return a list of dictionaries
            containing at least ``"energy"`` and ``"final_dump"`` keys. If not provided,
            fall back to calling ``gb_energy_func`` per candidate. If the function does
            not declare a ``checkpoint`` keyword argument it is automatically wrapped so
            that checkpointing still occurs at batch-return granularity; a
            :class:`~warnings.UserWarning` is emitted in that case. Declare ``checkpoint
            = None`` and call ``checkpoint.record(unique_id, energy, dump)`` per job to
            get per-job recovery granularity.
        :param crossover_surface: Keyword argument, optional, defaults to
            ``"periodic_wave"``. Formula-preserving crossover surface mode,
            ``"normal_plane"`` or ``"periodic_wave"``.
        :param crossover_max_tilt_degrees: Keyword argument, optional, defaults to
            ``5.0``. Maximum combined local periodic-wave tilt in degrees.
        :param crossover_attempts: Keyword argument, optional, defaults to ``8``.
            Maximum parent-pair attempts before one crossover slot falls back to
            mutation.
        :raises TypeError: If ``initial_ownership`` is not GrainOwnership, accompanies
            a non-file initial structure, ``allow_variable_cell`` is not Boolean, or a
            crossover policy argument has an invalid type.
        :raises ValueError: If ownership is supplied without an initial structure or
            variable-cell execution is requested without explicit ownership.
        """
        if not isinstance(allow_variable_cell, (bool, np.bool_)):
            raise TypeError("allow_variable_cell must be a Boolean")
        allow_variable_cell = bool(allow_variable_cell)
        if initial_ownership is not None:
            if not isinstance(initial_ownership, GrainOwnership):
                raise TypeError("initial_ownership must be a GrainOwnership instance")
            if initial_structure is None:
                raise ValueError("initial_ownership requires an initial_structure")
            if not isinstance(initial_structure, (str, Path)):
                raise TypeError(
                    "initial_ownership requires a str or Path initial_structure"
                )
        elif allow_variable_cell:
            raise ValueError("allow_variable_cell requires initial_ownership")
        if (
            isinstance(slice_and_merge_pct, (bool, np.bool_))
            or not isinstance(slice_and_merge_pct, Real)
        ):
            raise GBMinimizerTypeError(
                "slice_and_merge_pct must be a real number"
            )
        slice_and_merge_pct = float(slice_and_merge_pct)
        if not math.isfinite(slice_and_merge_pct) or not (
            0.0 <= slice_and_merge_pct <= 100.0
        ):
            raise GBMinimizerValueError(
                "slice_and_merge_pct must be finite and between 0 and 100"
            )
        if not isinstance(reuse_carryover_evaluations, (bool, np.bool_)):
            raise GBMinimizerTypeError(
                "reuse_carryover_evaluations must be a Boolean"
            )
        if not isinstance(crossover_surface, str):
            raise GBMinimizerTypeError("crossover_surface must be a string")
        if crossover_surface not in {"normal_plane", "periodic_wave"}:
            raise GBMinimizerValueError(
                "crossover_surface must be 'normal_plane' or 'periodic_wave'"
            )
        if (
            isinstance(crossover_max_tilt_degrees, (bool, np.bool_))
            or not isinstance(crossover_max_tilt_degrees, Real)
        ):
            raise GBMinimizerTypeError(
                "crossover_max_tilt_degrees must be a non-Boolean real scalar"
            )
        if (
            not np.isfinite(crossover_max_tilt_degrees)
            or float(crossover_max_tilt_degrees) < 0.0
            or float(crossover_max_tilt_degrees) >= 90.0
        ):
            raise GBMinimizerValueError(
                "crossover_max_tilt_degrees must be finite and satisfy 0 <= value < 90"
            )
        if (
            isinstance(crossover_attempts, (bool, np.bool_))
            or not isinstance(crossover_attempts, Integral)
        ):
            raise GBMinimizerTypeError(
                "crossover_attempts must be a non-Boolean integer"
            )
        if int(crossover_attempts) <= 0:
            raise GBMinimizerValueError(
                "crossover_attempts must be a positive integer"
            )
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
                        gb_batch_energy_func, penalty=ENERGY_PENALTY
                    )
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
                    gb_batch_energy_func, penalty=ENERGY_PENALTY
                )
            except TypeError:
                raise GBMinimizerTypeError(
                    "gb_batch_energy_func must be callable."
                )
        self.gb_batch_energy_func = gb_batch_energy_func
        self.history = []
        self.initial_structure = initial_structure
        self.initial_ownership = initial_ownership
        self.allow_variable_cell = allow_variable_cell
        self.local_random = np.random.default_rng(int(time()) if seed is None else seed)
        self._owned_evaluator = (
            ExplicitOwnershipEvaluator(
                GB=GB,
                scalar_energy_func=gb_energy_func,
                batch_energy_func=gb_batch_energy_func,
                local_random=self.local_random,
                penalty=ENERGY_PENALTY,
                allow_variable_cell=allow_variable_cell,
            )
            if initial_ownership is not None
            else None
        )
        self.manipulator = self._make_initial_manipulator()
        initial_parent = self.manipulator.parents[0]
        try:
            validate_formula_composition(
                initial_parent.whole_system,
                initial_parent.unit_cell,
            )
        except CandidateAdmissibilityError as exc:
            raise GBMinimizerValueError(
                f"initial candidate composition is inadmissible: {exc}"
            ) from exc
        self.composition_policy = tuple(initial_parent.unit_cell.formula_ratio)
        self.mutator = Mutator(choices, self.manipulator)
        self.manipulator.rng = self.local_random
        self.population_size = population_size
        self.generations = generations
        self.keep_top_pct = keep_top_pct
        self.intermediate_pct = intermediate_pct
        self.slice_and_merge_pct = slice_and_merge_pct
        self.reuse_carryover_evaluations = bool(reuse_carryover_evaluations)
        self.crossover_surface = crossover_surface
        self.crossover_max_tilt_degrees = float(crossover_max_tilt_degrees)
        self.crossover_attempts = int(crossover_attempts)
        self.GBE_vals = []

    def _make_initial_manipulator(self) -> GBManipulator:
        seed = self.initial_structure
        if seed is None:
            manip = GBManipulator(self.GB)
        elif isinstance(seed, GBMaker):
            manip = GBManipulator(seed)
        else:
            manip = GBManipulator(
                str(seed),
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
                grain_ownership=self.initial_ownership,
            )

        manip.rng = self.local_random

        return manip

    def _make_manipulator_from_file(self, filename: str) -> GBManipulator:
        if self.initial_ownership is not None:
            raise RuntimeError(
                "explicit-ownership file reloads must use reload_explicit_manipulator"
            )
        manipulator = GBManipulator(
            filename,
            unit_cell=self.GB.unit_cell,
            gb_thickness=self.GB.gb_thickness,
        )
        manipulator.rng = self.local_random
        return manipulator

    def _clone_owned_record(self, record: CandidateEvaluation) -> GBManipulator:
        """Clone a successfully reconstructed owned candidate.

        :param record: Successful explicit-ownership candidate evaluation.
        :return: Independent manipulator carrying the validated candidate state.
        :raises ValueError: If the evaluation did not produce a reusable candidate.
        """
        if (
            not record.success
            or record.manipulator is None
            or record.structure_path is None
        ):
            raise ValueError("cannot clone a failed candidate evaluation")
        manipulator = copy_module.copy(record.manipulator)
        manipulator.rng = self.local_random
        return manipulator

    def _make_next_owned_generation(
        self,
        records: list[CandidateEvaluation],
        intermediate_indices: list[int],
        offspring_count: int,
    ) -> tuple[list[GBManipulator], list[np.ndarray], list[list[str]]]:
        """Create exactly the requested number of ownership-aware offspring.

        :param records: Successful evaluations eligible for breeding.
        :param intermediate_indices: Indices eligible to become parents.
        :param offspring_count: Number of unfilled population slots.
        :return: Aligned manipulators, atom arrays, and lineages.
        :raises ValueError: If records are empty or ``offspring_count`` is invalid.
        """
        if not records:
            raise ValueError("no valid candidate records provided for breeding")
        if (
            isinstance(offspring_count, (bool, np.bool_))
            or not isinstance(offspring_count, Integral)
            or offspring_count < 0
        ):
            raise ValueError("offspring_count must be a nonnegative integer")
        offspring_count = int(offspring_count)
        if offspring_count == 0:
            return [], [], []
        if not intermediate_indices:
            intermediate_indices = list(range(len(records)))

        manipulators: list[GBManipulator] = []
        candidates: list[np.ndarray] = []
        lineages: list[list[str]] = []
        n_slice = math.floor(
            offspring_count * self.slice_and_merge_pct / 100.0
        )
        n_mutate = offspring_count - n_slice

        for _ in range(n_slice):
            failures: list[str] = []
            record1 = records[intermediate_indices[0]]
            crossed = False
            for _attempt in range(self.crossover_attempts):
                replace = len(intermediate_indices) < 2
                idx_1, idx_2 = self.local_random.choice(
                    intermediate_indices,
                    size=2,
                    replace=replace,
                )
                record1 = records[int(idx_1)]
                record2 = records[int(idx_2)]
                parent1 = self._clone_owned_record(record1).parents[0]
                parent2 = self._clone_owned_record(record2).parents[0]
                new_manipulator = GBManipulator._from_parents(
                    parent1,
                    parent2,
                    rng=self.local_random,
                )
                try:
                    new_structure = new_manipulator.slice_and_merge(
                        surface_mode=self.crossover_surface,
                        max_tilt_degrees=self.crossover_max_tilt_degrees,
                    )
                except CompositionAwareCrossoverError as exc:
                    failures.append(str(exc))
                    continue
                provenance = dict(new_manipulator.last_crossover_provenance or ())
                manipulators.append(new_manipulator)
                candidates.append(new_structure)
                lineages.append(
                    [
                        "slice_and_merge",
                        str(record1.structure_path),
                        str(record2.structure_path),
                        repr(provenance),
                    ]
                )
                crossed = True
                break
            if crossed:
                continue
            fallback = self._clone_owned_record(record1)
            mutation, new_structure = self.mutator.mutate(
                local_random=self.local_random,
                GB=self.GB,
                manipulator=fallback,
            )
            manipulators.append(fallback)
            candidates.append(new_structure)
            lineages.append(
                [
                    "crossover_fallback_" + mutation,
                    str(record1.structure_path),
                    f"{len(failures)} inadmissible crossover attempts",
                ]
            )

        if n_mutate:
            selected = self.local_random.choice(
                intermediate_indices,
                size=n_mutate,
                replace=True,
            )
            for idx in selected:
                record = records[int(idx)]
                new_manipulator = self._clone_owned_record(record)
                mutation, new_structure = self.mutator.mutate(
                    local_random=self.local_random,
                    GB=self.GB,
                    manipulator=new_manipulator,
                )
                manipulators.append(new_manipulator)
                candidates.append(new_structure)
                lineages.append([mutation, str(record.structure_path)])

        return manipulators, candidates, lineages

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
        gen_checkpoint: CandidateCheckpoint | None = None,
        cached_evaluations: list[_CachedEvaluation | None] | None = None,
    ) -> tuple[list[float], list[str | None], list[GBManipulator | None]]:
        """Evaluate all candidates, optionally using a batch energy function.

        :param gen_checkpoint: If provided, already-evaluated candidates are skipped
            and new results are recorded after each evaluation.
        :param cached_evaluations: Successful results aligned to unchanged carryover
            candidates. ``None`` entries are evaluated normally.
        :return: Aligned energies, evaluator artifact paths, and manipulators.
        :raises ValueError: If cached results are not population-aligned.
        """

        population_length = len(population_structures)
        if cached_evaluations is None:
            cached_evaluations = [None] * population_length
        elif len(cached_evaluations) != population_length:
            raise ValueError("cached evaluations must remain population-aligned")

        all_uids = [
            f"GA_{unique_id}_g{gen}_c{i}"
            for i in range(len(population_structures))
        ]

        if self.gb_batch_energy_func is not None:
            batch_results: list[dict[str, object] | None] = [
                None
            ] * population_length
            pending = []
            for index, uid in enumerate(all_uids):
                cached = cached_evaluations[index]
                if cached is not None and self._is_valid_file(
                    cached.structure_path
                ):
                    batch_results[index] = {
                        "energy": cached.energy,
                        "final_dump": cached.structure_path,
                    }
                elif gen_checkpoint is None or not gen_checkpoint.is_done(uid):
                    pending.append((index, uid))

            if gen_checkpoint is not None:
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
                for index, uid in enumerate(all_uids):
                    if batch_results[index] is not None:
                        continue
                    energy, final_dump = gen_checkpoint.get_result(uid)
                    batch_results[index] = {
                        "energy": energy,
                        "final_dump": final_dump,
                    }
            else:
                if pending:
                    pending_idxs, pending_uids = zip(*pending)
                    raw_results = self.gb_batch_energy_func(
                        self.GB,
                        [population_manipulators[i] for i in pending_idxs],
                        [population_structures[i] for i in pending_idxs],
                        [population_lineages[i] for i in pending_idxs],
                        list(pending_uids),
                    )
                    for index, result in zip(
                        pending_idxs,
                        raw_results,
                        strict=True,
                    ):
                        batch_results[index] = result

            gen_energies = []
            gen_files = []
            evaluated_manipulators = []
            for result in batch_results:
                if result is None:
                    raise RuntimeError("batch evaluation lost candidate alignment")
                energy = float(result.get("energy", ENERGY_PENALTY))
                dump = result.get("final_dump", None)

                gen_energies.append(energy)
                if self._is_valid_file(dump):
                    gen_files.append(dump)
                    try:
                        evaluated_manipulators.append(
                            self._make_manipulator_from_file(dump)
                        )
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
        gen_files: list[str | None] = []
        evaluated_manipulators: list[GBManipulator | None] = []

        for idx, (manipulator, atom_positions) in enumerate(
                zip(population_manipulators, population_structures)):
            uid = all_uids[idx]
            cached = cached_evaluations[idx]
            if cached is not None and self._is_valid_file(cached.structure_path):
                gbe = cached.energy
                dump_file_name = cached.structure_path
            elif gen_checkpoint is not None and gen_checkpoint.is_done(uid):
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
                        self._make_manipulator_from_file(dump_file_name)
                    )
                except Exception:
                    gen_files[-1] = None
                    gen_energies[-1] = ENERGY_PENALTY
                    evaluated_manipulators.append(None)
            else:
                gen_files.append(None)
                gen_energies[-1] = ENERGY_PENALTY
                evaluated_manipulators.append(None)

        return gen_energies, gen_files, evaluated_manipulators

    def _make_next_generation(
        self,
        files: list[str],
        intermediate_indices: list[int],
        offspring_count: int,
    ) -> tuple[list[GBManipulator], list[np.ndarray], list[list[str]]]:
        """Create exactly the requested number of legacy-path offspring.

        :param files: Valid evaluated structure files eligible for breeding.
        :param intermediate_indices: Indices eligible to become parents.
        :param offspring_count: Number of unfilled population slots.
        :return: Aligned manipulators, atom arrays, and lineages.
        :raises ValueError: If no parent files are provided or ``offspring_count`` is
            invalid.
        """
        if not files:
            raise ValueError(
                "No valid parent files provided to _make_next_generation()."
            )
        if (
            isinstance(offspring_count, (bool, np.bool_))
            or not isinstance(offspring_count, Integral)
            or offspring_count < 0
        ):
            raise ValueError("offspring_count must be a nonnegative integer")
        offspring_count = int(offspring_count)
        if offspring_count == 0:
            return [], [], []

        if not intermediate_indices:
            intermediate_indices = list(range(len(files)))
        candidates: list[np.ndarray] = []
        manipulators: list[GBManipulator] = []
        lineages: list[list[str]] = []

        N_slice = math.floor(
            offspring_count * self.slice_and_merge_pct / 100.0
        )
        N_mutate = offspring_count - N_slice

        # Slice & merge
        for _ in range(N_slice):
            p1 = files[intermediate_indices[0]]
            crossed = False
            for _attempt in range(self.crossover_attempts):
                replace = len(intermediate_indices) < 2
                idx_1, idx_2 = self.local_random.choice(
                    intermediate_indices,
                    size=2,
                    replace=replace,
                )
                p1, p2 = files[int(idx_1)], files[int(idx_2)]
                new_manip = GBManipulator(
                    p1,
                    p2,
                    unit_cell=self.GB.unit_cell,
                    gb_thickness=self.GB.gb_thickness,
                )
                new_manip.rng = self.local_random
                try:
                    new_struct = new_manip.slice_and_merge(
                        surface_mode=self.crossover_surface,
                        max_tilt_degrees=self.crossover_max_tilt_degrees,
                    )
                except CompositionAwareCrossoverError:
                    continue
                candidates.append(new_struct)
                manipulators.append(new_manip)
                lineages.append(
                    [
                        "slice_and_merge",
                        p1,
                        p2,
                        repr(dict(new_manip.last_crossover_provenance or ())),
                    ]
                )
                crossed = True
                break
            if crossed:
                continue
            fallback = self._make_manipulator_from_file(p1)
            mutation, new_struct = self.mutator.mutate(
                local_random=self.local_random,
                GB=self.GB,
                manipulator=fallback,
            )
            candidates.append(new_struct)
            manipulators.append(fallback)
            lineages.append(["crossover_fallback_" + mutation, p1])

        # Mutations
        if not intermediate_indices:
            intermediate_indices = list(range(len(files)))
        choices = self.local_random.choice(
            intermediate_indices, size=N_mutate, replace=True
        )
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

    def _is_valid_file(self, p: str | None) -> bool:
        return bool(p) and Path(p).is_file()

    @staticmethod
    def _cached_evaluation_to_state(
        record: _CachedEvaluation | None,
    ) -> dict | None:
        """Serialize one optional legacy carryover cache entry.

        :param record: Reusable result or ``None`` for a cache miss.
        :return: JSON-safe cache state.
        """
        if record is None:
            return None
        return {
            "energy": record.energy,
            "structure_path": record.structure_path,
        }

    @staticmethod
    def _cached_evaluation_from_state(state: object) -> _CachedEvaluation | None:
        """Restore one optional legacy carryover cache entry.

        :param state: Deserialized optional cache state.
        :return: Validated reusable result or ``None``.
        :raises GBMinimizerError: If cache state is malformed.
        """
        if state is None:
            return None
        if not isinstance(state, dict):
            raise GBMinimizerError("cached evaluation state must be a dictionary")
        try:
            energy = float(state["energy"])
            structure_path = state["structure_path"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GBMinimizerError("cached evaluation state is malformed") from exc
        if not math.isfinite(energy):
            raise GBMinimizerError("cached evaluation energy must be finite")
        if not isinstance(structure_path, str) or not structure_path:
            raise GBMinimizerError("cached evaluation structure_path is invalid")
        return _CachedEvaluation(energy=energy, structure_path=structure_path)

    @staticmethod
    def _owned_evaluation_to_state(record: CandidateEvaluation) -> dict:
        """Serialize one typed owned evaluation without its live manipulator.

        :param record: Explicit-ownership evaluation to persist.
        :return: JSON-safe evaluation state.
        """
        return {
            "candidate_id": record.candidate_id,
            "input_index": record.input_index,
            "energy": record.energy,
            "structure_path": record.structure_path,
            "mapping": (
                None
                if record.mapping is None
                else _candidate_mapping_to_state(record.mapping)
            ),
            "success": record.success,
            "failure_reason": record.failure_reason,
        }

    def _owned_evaluation_from_state(self, state: object) -> CandidateEvaluation:
        """Reconstruct one typed owned evaluation from checkpoint state.

        Successful artifacts are reloaded through the authoritative explicit-ownership
        path. Failed records remain non-reusable and do not require their diagnostic
        artifact to exist.

        :param state: Deserialized evaluation state.
        :return: Validated typed evaluation.
        :raises GBMinimizerError: If the state or a required successful artifact is
            invalid.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned evaluation restore requires an evaluator adapter"
            )
        if not isinstance(state, dict):
            raise GBMinimizerError("owned evaluation state must be a dictionary")
        try:
            candidate_id = state["candidate_id"]
            input_index = int(state["input_index"])
            energy = float(state["energy"])
            structure_path = state["structure_path"]
            success = state["success"]
            failure_reason = state.get("failure_reason")
            mapping_state = state["mapping"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GBMinimizerError("owned evaluation state is malformed") from exc
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise GBMinimizerError("owned evaluation candidate_id is invalid")
        if isinstance(state.get("input_index"), (bool, np.bool_)) or input_index < -1:
            raise GBMinimizerError("owned evaluation input_index is invalid")
        if not np.isfinite(energy):
            raise GBMinimizerError("owned evaluation energy must be finite")
        if not isinstance(success, bool):
            raise GBMinimizerError("owned evaluation success must be Boolean")
        if structure_path is not None and not isinstance(structure_path, str):
            raise GBMinimizerError("owned evaluation structure_path is invalid")
        try:
            mapping = (
                None
                if mapping_state is None
                else _candidate_mapping_from_state(mapping_state)
            )
        except GrainOwnershipError as exc:
            raise GBMinimizerError(
                f"owned evaluation mapping is invalid: {exc}"
            ) from exc
        if not success:
            if not isinstance(failure_reason, str) or not failure_reason:
                raise GBMinimizerError(
                    "failed owned evaluation lacks failure context"
                )
            if energy != self._owned_evaluator.penalty:
                raise GBMinimizerError(
                    "failed owned evaluation does not carry the configured penalty"
                )
            return CandidateEvaluation(
                candidate_id=candidate_id,
                input_index=input_index,
                energy=energy,
                structure_path=structure_path,
                mapping=mapping,
                manipulator=None,
                success=False,
                failure_reason=failure_reason,
            )
        if mapping is None or structure_path is None:
            raise GBMinimizerError(
                "successful owned evaluation lacks reconstruction state"
            )
        try:
            manipulator = self._owned_evaluator._reload_mapping(
                structure_path,
                mapping,
            )
        except (
            OSError,
            LammpsDataError,
            GrainOwnershipError,
            ParentError,
            GBManipulatorError,
        ) as exc:
            raise GBMinimizerError(
                "Checkpoint owned evaluation artifact is missing, unreadable, or "
                f"inconsistent: {structure_path}"
            ) from exc
        return CandidateEvaluation(
            candidate_id=candidate_id,
            input_index=input_index,
            energy=energy,
            structure_path=structure_path,
            mapping=mapping,
            manipulator=manipulator,
            success=True,
        )

    def _write_owned_population_checkpoint(
        self,
        checkpoint_file: Path,
        unique_id: str,
        next_generation: int,
        manipulators: list[GBManipulator],
        structures: list[np.ndarray],
    ) -> list[dict]:
        """Write owned pending structures and their explicit reconstruction metadata.

        :param checkpoint_file: Run-level checkpoint path whose directory owns artifacts.
        :param unique_id: Stable run identifier.
        :param next_generation: Generation that will consume the pending population.
        :param manipulators: Candidate manipulators in population order.
        :param structures: Candidate atom rows in matching population order.
        :return: Ordered serialized population snapshots.
        :raises GBMinimizerError: If population alignment or ownership is invalid.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned population checkpoint requires an evaluator adapter"
            )
        if len(manipulators) != len(structures):
            raise GBMinimizerError(
                "owned checkpoint population lost manipulator/structure alignment"
            )
        snapshots = []
        for index, (manipulator, structure) in enumerate(
            zip(manipulators, structures, strict=True)
        ):
            try:
                mapping = self._owned_evaluator._candidate_file_mapping(
                    manipulator,
                    structure,
                )
            except GrainOwnershipError as exc:
                raise GBMinimizerError(
                    f"owned checkpoint candidate {index} has invalid ownership: {exc}"
                ) from exc
            pending_path = checkpoint_file.parent / (
                f"GA_{unique_id}_g{next_generation}_c{index}.owned.pending"
            )
            try:
                self.GB.write_lammps(
                    str(pending_path),
                    structure,
                    mapping.box_dims,
                    precision=15,
                )
            except (OSError, GBMakerError) as exc:
                raise GBMinimizerError(
                    f"could not persist owned checkpoint candidate {index}"
                ) from exc
            snapshots.append(
                {
                    "structure_path": str(pending_path),
                    "mapping": _candidate_mapping_to_state(mapping),
                }
            )
        return snapshots

    def _restore_owned_population(
        self,
        snapshots: object,
    ) -> tuple[list[GBManipulator], list[np.ndarray]]:
        """Restore an aligned pending owned population from checkpoint snapshots.

        :param snapshots: Ordered serialized structure/mapping snapshots.
        :return: Reconstructed manipulators and atom arrays.
        :raises GBMinimizerError: If state is malformed or any required artifact fails
            explicit reload validation.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned population restore requires an evaluator adapter"
            )
        if not isinstance(snapshots, list) or len(snapshots) != self.population_size:
            raise GBMinimizerError(
                "owned checkpoint population has an invalid candidate count"
            )
        manipulators = []
        structures = []
        for index, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, dict):
                raise GBMinimizerError(
                    f"owned checkpoint candidate {index} is malformed"
                )
            path = snapshot.get("structure_path")
            if not isinstance(path, str):
                raise GBMinimizerError(
                    f"owned checkpoint candidate {index} lacks a structure path"
                )
            try:
                mapping = _candidate_mapping_from_state(snapshot.get("mapping"))
                manipulator = self._owned_evaluator._reload_mapping(path, mapping)
            except (
                OSError,
                LammpsDataError,
                GrainOwnershipError,
                ParentError,
                GBManipulatorError,
            ) as exc:
                raise GBMinimizerError(
                    f"Checkpoint owned population path {path} is missing, unreadable, "
                    "or inconsistent."
                ) from exc
            manipulators.append(manipulator)
            structures.append(
                np.array(manipulator.parents[0].whole_system, copy=True)
            )
        return manipulators, structures

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

        Checkpointing is optional. Pass ``checkpoint_file`` to enable it; omit it (or
        pass ``None``) to run without any checkpoint file. When enabled, a per-candidate
        sidecar(``{stem}.iter{N}{ext}``) is also written so a mid-generation crash can
        be resumed without re-evaluating completed candidates. The checkpoint file is
        **not** deleted on normal completion — it can be used to continue the run later
        by calling ``run_GA`` again with the same ``checkpoint_file`` after increasing
        ``generations``. The checkpoint file and the sibling ``*.pending`` structure
        files in the same directory form a unit - both must be present to resume or
        extend a run. Do not delete or move the ``.pending`` files independently of the
        checkpoint file.

        :param unique_id: Argument, optional, defaults to ``None``. Label applied to all
            output files. Restored from the checkpoint on resume if not provided.
        :param checkpoint_file: Keyword argument, optional, defaults to ``None``. Path to
            the run-level checkpoint file. If the file exists the run resumes from it;
            otherwise a fresh run begins and the file is created.
        :param checkpoint_format: Keyword argument, optional, defaults to ``"json"``.
            Serialization format: ``"json"`` (human-readable) or ``"pickle"`` (binary,
            no NumPy conversion needed).
        :param checkpoint_interval: Keyword argument, optional, defaults to 1. Save a
            run-level checkpoint every N generations.
        :return: Tuple containing the minimum energy value observed and the associated
            dump filename.
        :raises GBMinimizerError: If a checkpoint is malformed or references a missing,
            unreadable, or ownership-inconsistent required structure artifact.
        :raises GBMinimizerValueError: If checkpoint configuration is invalid.
        """

        if self.initial_ownership is not None:
            return self._run_owned_GA(
                unique_id=unique_id,
                checkpoint_file=checkpoint_file,
                checkpoint_format=checkpoint_format,
                checkpoint_interval=checkpoint_interval,
            )

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
                    saved_slice_pct = state["run_params"].get(
                        "slice_and_merge_pct",
                        50.0,
                    )
                    if saved_slice_pct != self.slice_and_merge_pct:
                        raise GBMinimizerError(
                            "checkpoint slice_and_merge_pct does not match the "
                            "minimizer configuration"
                        )
                    saved_reuse = state["run_params"].get(
                        "reuse_carryover_evaluations",
                        False,
                    )
                    if saved_reuse != self.reuse_carryover_evaluations:
                        raise GBMinimizerError(
                            "checkpoint reuse_carryover_evaluations does not match "
                            "the minimizer configuration"
                        )
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
            cached_states = state["state"].get(
                "population_cached_evaluations",
                [None] * self.population_size,
            )
            if not isinstance(cached_states, list) or len(cached_states) != len(
                population_lineages
            ):
                raise GBMinimizerError(
                    "checkpoint cached evaluations are not population-aligned"
                )
            population_cached_evaluations = [
                self._cached_evaluation_from_state(cached_state)
                for cached_state in cached_states
            ]
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
                    np.array(seed_manip.parents[0].whole_system, copy=True)
                )
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
            population_cached_evaluations = [None] * self.population_size
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
                "run_params": {
                    "unique_id": str(unique_id),
                    "slice_and_merge_pct": self.slice_and_merge_pct,
                    "reuse_carryover_evaluations": (
                        self.reuse_carryover_evaluations
                    ),
                },
                "state": {
                    "GBE_vals": self.GBE_vals,
                    "history": self.history,
                    "population_lineages": population_lineages,
                    "population_checkpoint_paths": population_checkpoint_paths,
                    "population_cached_evaluations": [
                        self._cached_evaluation_to_state(record)
                        for record in population_cached_evaluations
                    ],
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
                cached_evaluations=population_cached_evaluations,
            )

            valid_old_idxs = [
                i for i, f in enumerate(gen_files) if self._is_valid_file(f)
            ]

            self.GBE_vals.append(gen_energies)
            self.history.append(list(zip(population_lineages, gen_energies)))

            if not valid_old_idxs:
                # If nothing valid survived evaluation, re-seed from best.
                next_manipulators = []
                next_structures = []
                next_lineages = []
                next_cached_evaluations: list[_CachedEvaluation | None] = []

                for _ in range(self.population_size):
                    candidate_manip = self._make_manipulator_from_file(
                        best_dump
                    )
                    mutation, candidate_struct = self.mutator.mutate(
                        local_random=self.local_random,
                        GB=self.GB,
                        manipulator=candidate_manip,
                    )
                    next_manipulators.append(candidate_manip)
                    next_structures.append(candidate_struct)
                    next_lineages.append([mutation, best_dump])
                    next_cached_evaluations.append(None)

                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
                population_cached_evaluations = next_cached_evaluations
            else:
                for i in valid_old_idxs:
                    gbe = gen_energies[i]
                    dump_file_name = gen_files[i]
                    if gbe < best_energy:
                        best_energy = gbe
                        best_dump = dump_file_name

                # Build compressed arrays of only valid candidates for selection and breeding.
                valid_energies = [gen_energies[i] for i in valid_old_idxs]
                valid_files = [gen_files[i] for i in valid_old_idxs]

                lowest_valid_idxs, inter_valid_idxs = self._select_indices_by_energy(
                    valid_energies
                )

                # Carry over lowest energies.
                next_manipulators = []
                next_structures = []
                next_lineages = []
                next_cached_evaluations = []
                for j in lowest_valid_idxs:
                    old_idx = valid_old_idxs[j]
                    manip = evaluated_manipulators[old_idx]
                    dump = gen_files[old_idx]
                    if manip is None or dump is None:
                        continue
                    next_manipulators.append(manip)
                    next_structures.append(manip.parents[0].whole_system)
                    next_lineages.append(["carryover", dump])
                    next_cached_evaluations.append(
                        _CachedEvaluation(gen_energies[old_idx], dump)
                        if self.reuse_carryover_evaluations
                        else None
                    )

                valid_files_str = [f for f in valid_files if f is not None]
                offspring_count = self.population_size - len(next_manipulators)
                new_manips, new_structs, new_lineages = self._make_next_generation(
                    valid_files_str,
                    inter_valid_idxs,
                    offspring_count,
                )

                next_manipulators.extend(new_manips)
                next_structures.extend(new_structs)
                next_lineages.extend(new_lineages)
                next_cached_evaluations.extend([None] * len(new_lineages))

                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
                population_cached_evaluations = next_cached_evaluations

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

    def _run_owned_GA(
        self,
        unique_id: int | uuid.UUID | None = None,
        *,
        checkpoint_file: str | Path | None = None,
        checkpoint_format: str = "json",
        checkpoint_interval: int = 1,
    ) -> tuple[float, str]:
        """Run the GA while preserving explicit ownership through every reload.

        :param unique_id: Argument, optional, defaults to ``None``. Run identifier.
        :param checkpoint_file: Keyword argument, optional, defaults to ``None``. Run-level
            checkpoint path used for generation-boundary and candidate-sidecar recovery.
        :param checkpoint_format: Keyword argument, optional, defaults to ``"json"``.
            Checkpoint serialization format, either ``"json"`` or ``"pickle"``.
        :param checkpoint_interval: Keyword argument, optional, defaults to 1. Save the
            run-level checkpoint every N completed generations.
        :return: Minimum energy and validated structure path.
        :raises GBMinimizerError: If evaluation fails initially, aligned population state
            cannot be maintained, or checkpoint state cannot be reconstructed safely.
        :raises GBMinimizerValueError: If checkpoint configuration is invalid.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "explicit-ownership execution requires an evaluator adapter"
            )

        try:
            if checkpoint_file is None:
                checkpoint = CheckpointStore.disabled()
                state = None
                unique_id = str(unique_id) if unique_id is not None else str(
                    uuid.uuid4())
            else:
                checkpoint_file = Path(checkpoint_file)
                checkpoint = CheckpointStore.from_optional(
                    checkpoint_file,
                    checkpoint_format,
                    checkpoint_interval,
                )
                state = checkpoint.load()
                if state is None:
                    unique_id = (
                        str(unique_id) if unique_id is not None else str(uuid.uuid4())
                    )
                else:
                    unique_id = state["run_params"]["unique_id"]
        except CheckpointError as exc:
            raise GBMinimizerValueError(str(exc)) from exc
        except (KeyError, TypeError) as exc:
            raise GBMinimizerError(
                "Invalid explicit-ownership GA checkpoint envelope."
            ) from exc

        self._owned_evaluator.begin_run()

        population_snapshots: list[dict] = []
        if state is not None:
            try:
                if not isinstance(state, dict):
                    raise GBMinimizerError(
                        "checkpoint envelope must be a dictionary"
                    )
                if (
                    state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                    or state.get("minimizer") != "GeneticAlgorithmMinimizer"
                    or state.get("progress_unit") != "generation"
                ):
                    raise GBMinimizerError(
                        "checkpoint envelope is not a supported genetic-algorithm state"
                    )
                owned_state = state["state"]
                if (
                    owned_state.get("ga_mode") != "explicit_ownership"
                    or owned_state.get("owned_checkpoint_version")
                    != _OWNED_GA_CHECKPOINT_VERSION
                ):
                    raise GBMinimizerError(
                        "checkpoint does not contain supported explicit-ownership state"
                    )
                run_params = state["run_params"]
                expected_params = {
                    "population_size": self.population_size,
                    "keep_top_pct": self.keep_top_pct,
                    "intermediate_pct": self.intermediate_pct,
                    "slice_and_merge_pct": self.slice_and_merge_pct,
                    "reuse_carryover_evaluations": (
                        self.reuse_carryover_evaluations
                    ),
                    "allow_variable_cell": self.allow_variable_cell,
                    "choices": self.mutator.choices_keys,
                    "crossover_surface": self.crossover_surface,
                    "crossover_max_tilt_degrees": (
                        self.crossover_max_tilt_degrees
                    ),
                    "crossover_attempts": self.crossover_attempts,
                    "composition_policy": [
                        [species, coefficient]
                        for species, coefficient in self.composition_policy
                    ],
                }
                parameter_defaults = {
                    "slice_and_merge_pct": 50.0,
                    "reuse_carryover_evaluations": False,
                }
                for name, expected in expected_params.items():
                    default = parameter_defaults.get(name)
                    if run_params.get(name, default) != expected:
                        raise GBMinimizerError(
                            f"owned checkpoint run parameter {name!r} does not match "
                            "the minimizer configuration"
                        )

                progress_index = state["progress_index"]
                if (
                    isinstance(progress_index, (bool, np.bool_))
                    or not isinstance(progress_index, Integral)
                    or progress_index < 0
                ):
                    raise GBMinimizerError(
                        "owned checkpoint progress_index is invalid"
                    )
                self.GBE_vals = owned_state["GBE_vals"]
                self.history = owned_state["history"]
                if (
                    not isinstance(self.GBE_vals, list)
                    or len(self.GBE_vals) != progress_index + 2
                    or not isinstance(self.history, list)
                    or len(self.history) != progress_index + 1
                ):
                    raise GBMinimizerError(
                        "owned checkpoint energy/history progress is inconsistent"
                    )
                self.local_random.bit_generator.state = state["rng_state"]
                _start_gen = int(progress_index) + 1
                best_record = self._owned_evaluation_from_state(
                    owned_state["best_evaluation"]
                )
                if not best_record.success:
                    raise GBMinimizerError(
                        "owned checkpoint best evaluation is not reusable"
                    )
                if not np.isclose(
                    best_record.energy,
                    float(state["best_energy"]),
                    rtol=0.0,
                    atol=0.0,
                ) or best_record.structure_path != state["best_dump"]:
                    raise GBMinimizerError(
                        "owned checkpoint best-evaluation envelope is inconsistent"
                    )
                population_lineages = owned_state["population_lineages"]
                if (
                    not isinstance(population_lineages, list)
                    or len(population_lineages) != self.population_size
                    or not all(isinstance(lineage, list) for lineage in population_lineages)
                ):
                    raise GBMinimizerError(
                        "owned checkpoint population lineages are invalid"
                    )
                population_snapshots = owned_state["population_candidates"]
                population_manipulators, population_structures = (
                    self._restore_owned_population(population_snapshots)
                )
                cached_states = owned_state.get(
                    "population_cached_evaluations",
                    [None] * self.population_size,
                )
                if not isinstance(cached_states, list) or len(
                    cached_states
                ) != len(population_manipulators):
                    raise GBMinimizerError(
                        "owned checkpoint cached evaluations are not "
                        "population-aligned"
                    )
                population_cached_evaluations = [
                    None
                    if cached_state is None
                    else self._owned_evaluation_from_state(cached_state)
                    for cached_state in cached_states
                ]
                last_states = owned_state["last_generation_evaluations"]
                if not isinstance(last_states, list) or len(last_states) != self.population_size:
                    raise GBMinimizerError(
                        "owned checkpoint generation evaluations are invalid"
                    )
                self.last_generation_evaluations = [
                    self._owned_evaluation_from_state(record_state)
                    for record_state in last_states
                ]
                self._owned_evaluator.restore_claimed_paths(
                    owned_state["claimed_paths"]
                )
                self.best_evaluation = best_record
                stale = CandidateCheckpoint._derive_path(
                    checkpoint_file,
                    int(progress_index),
                )
                if stale.exists():
                    stale.unlink()
            except GBMinimizerError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise GBMinimizerError(
                    f"Invalid explicit-ownership GA checkpoint state: {exc}"
                ) from exc
        else:
            self.GBE_vals = []
            self.history = []
            self.last_generation_evaluations = []
            initial_atoms = np.array(
                self.manipulator.parents[0].whole_system,
                copy=True,
            )
            # No mutation has occurred yet, so initial labels are the persistent labels
            # carried by the owned parent.
            initial_record = self._owned_evaluator.evaluate_candidate(
                self.manipulator,
                initial_atoms,
                f"GA_initial{unique_id}",
                -1,
            )
            if not initial_record.success or initial_record.structure_path is None:
                raise GBMinimizerError(
                    "initial explicit-ownership evaluation failed: "
                    f"{initial_record.failure_reason}"
                )
            self.GBE_vals.append([initial_record.energy])
            best_record = initial_record
            self.best_evaluation = best_record

            population_manipulators = []
            population_structures = []
            population_lineages = []
            population_cached_evaluations: list[
                CandidateEvaluation | None
            ] = []
            seed_manipulator = self._clone_owned_record(initial_record)
            population_manipulators.append(seed_manipulator)
            population_structures.append(
                np.array(seed_manipulator.parents[0].whole_system, copy=True)
            )
            population_lineages.append(["START", initial_record.structure_path])
            population_cached_evaluations.append(None)

            for _ in range(self.population_size - 1):
                candidate_manipulator = self._clone_owned_record(initial_record)
                mutation, candidate_structure = self.mutator.mutate(
                    local_random=self.local_random,
                    GB=self.GB,
                    manipulator=candidate_manipulator,
                )
                population_manipulators.append(candidate_manipulator)
                population_structures.append(candidate_structure)
                population_lineages.append([mutation, initial_record.structure_path])
                population_cached_evaluations.append(None)
            _start_gen = 0

        def _build_owned_state(gen: int) -> dict:
            return {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "minimizer": "GeneticAlgorithmMinimizer",
                "progress_unit": "generation",
                "progress_index": gen,
                "best_energy": best_record.energy,
                "best_dump": best_record.structure_path,
                "rng_state": self.local_random.bit_generator.state,
                "run_params": {
                    "unique_id": str(unique_id),
                    "population_size": self.population_size,
                    "keep_top_pct": self.keep_top_pct,
                    "intermediate_pct": self.intermediate_pct,
                    "slice_and_merge_pct": self.slice_and_merge_pct,
                    "reuse_carryover_evaluations": (
                        self.reuse_carryover_evaluations
                    ),
                    "allow_variable_cell": self.allow_variable_cell,
                    "choices": self.mutator.choices_keys,
                    "crossover_surface": self.crossover_surface,
                    "crossover_max_tilt_degrees": (
                        self.crossover_max_tilt_degrees
                    ),
                    "crossover_attempts": self.crossover_attempts,
                    "composition_policy": [
                        [species, coefficient]
                        for species, coefficient in self.composition_policy
                    ],
                },
                "state": {
                    "ga_mode": "explicit_ownership",
                    "owned_checkpoint_version": _OWNED_GA_CHECKPOINT_VERSION,
                    "GBE_vals": self.GBE_vals,
                    "history": self.history,
                    "population_lineages": population_lineages,
                    "population_candidates": population_snapshots,
                    "population_cached_evaluations": [
                        None
                        if record is None
                        else self._owned_evaluation_to_state(record)
                        for record in population_cached_evaluations
                    ],
                    "best_evaluation": self._owned_evaluation_to_state(best_record),
                    "last_generation_evaluations": [
                        self._owned_evaluation_to_state(record)
                        for record in self.last_generation_evaluations
                    ],
                    "claimed_paths": self._owned_evaluator.claimed_paths_state(),
                },
            }

        for gen in range(_start_gen, self.generations):
            current_pending = [
                snapshot["structure_path"]
                for snapshot in population_snapshots
                if str(snapshot.get("structure_path", "")).endswith(
                    ".owned.pending"
                )
            ]
            all_uids = [
                f"GA_{unique_id}_g{gen}_c{index}"
                for index in range(len(population_structures))
            ]
            try:
                gen_checkpoint = (
                    CandidateCheckpoint.new_or_resume(
                        checkpoint_file,
                        checkpoint_format,
                        gen,
                        all_uids,
                    )
                    if checkpoint.enabled
                    else None
                )
                records = self._owned_evaluator.evaluate_generation(
                    population_manipulators,
                    population_structures,
                    population_lineages,
                    gen,
                    unique_id,
                    gen_checkpoint=gen_checkpoint,
                    cached_evaluations=population_cached_evaluations,
                )
            except CheckpointError as exc:
                raise GBMinimizerError(str(exc)) from exc
            self.last_generation_evaluations = records
            generation_energies = [record.energy for record in records]
            self.GBE_vals.append(generation_energies)
            self.history.append(list(zip(population_lineages, generation_energies)))
            valid_records = [record for record in records if record.success]

            if not valid_records:
                next_manipulators: list[GBManipulator] = []
                next_structures: list[np.ndarray] = []
                next_lineages: list[list[str]] = []
                next_cached_evaluations: list[
                    CandidateEvaluation | None
                ] = []
                for _ in range(self.population_size):
                    candidate_manipulator = self._clone_owned_record(best_record)
                    mutation, candidate_structure = self.mutator.mutate(
                        local_random=self.local_random,
                        GB=self.GB,
                        manipulator=candidate_manipulator,
                    )
                    next_manipulators.append(candidate_manipulator)
                    next_structures.append(candidate_structure)
                    next_lineages.append([mutation, best_record.structure_path])
                    next_cached_evaluations.append(None)
                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
                population_cached_evaluations = next_cached_evaluations
            else:
                for record in valid_records:
                    if record.energy < best_record.energy:
                        best_record = record
                        self.best_evaluation = record

                valid_energies = [record.energy for record in valid_records]
                lowest_indices, intermediate_indices = self._select_indices_by_energy(
                    valid_energies
                )
                next_manipulators = []
                next_structures = []
                next_lineages = []
                next_cached_evaluations = []
                for index in lowest_indices:
                    record = valid_records[index]
                    carryover = self._clone_owned_record(record)
                    next_manipulators.append(carryover)
                    next_structures.append(
                        np.array(carryover.parents[0].whole_system, copy=True)
                    )
                    next_lineages.append(["carryover", record.structure_path])
                    next_cached_evaluations.append(
                        record if self.reuse_carryover_evaluations else None
                    )

                offspring_count = self.population_size - len(next_manipulators)
                new_manipulators, new_structures, new_lineages = (
                    self._make_next_owned_generation(
                        valid_records,
                        intermediate_indices,
                        offspring_count,
                    )
                )
                next_manipulators.extend(new_manipulators)
                next_structures.extend(new_structures)
                next_lineages.extend(new_lineages)
                next_cached_evaluations.extend([None] * len(new_lineages))
            if not (
                len(next_manipulators)
                == len(next_structures)
                == len(next_lineages)
                == self.population_size
            ):
                raise GBMinimizerError(
                    "owned GA failed to construct a complete aligned population"
                )
            population_manipulators = next_manipulators
            population_structures = next_structures
            population_lineages = next_lineages
            population_cached_evaluations = next_cached_evaluations

            is_final_gen = gen == self.generations - 1
            if checkpoint.enabled and (checkpoint.is_due(gen + 1) or is_final_gen):
                new_snapshots = self._write_owned_population_checkpoint(
                    checkpoint_file,
                    str(unique_id),
                    gen + 1,
                    population_manipulators,
                    population_structures,
                )
                population_snapshots = new_snapshots
                try:
                    checkpoint.save_final(_build_owned_state(gen))
                except CheckpointError as exc:
                    raise GBMinimizerError(str(exc)) from exc
                for path in current_pending:
                    Path(path).unlink(missing_ok=True)

            # Candidate sidecars are transient once the generation boundary is safely
            # represented by the main checkpoint (or checkpointing is disabled).
            if gen_checkpoint is not None:
                gen_checkpoint.delete()

        self.best_evaluation = best_record
        return best_record.energy, str(best_record.structure_path)
