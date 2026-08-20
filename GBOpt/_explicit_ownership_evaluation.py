"""Evaluator adaptation for file-backed candidates with explicit grain ownership.

This module owns callback invocation, result normalization, artifact validation, and
candidate reconstruction. Optimizer selection and breeding policy do not belong here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from uuid import UUID

import numpy as np

from GBOpt._candidate_admissibility import (
    CandidateAdmissibilityError,
    validate_formula_composition,
)
from GBOpt.Checkpoint import CandidateCheckpoint
from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnershipError,
    LammpsDataError,
    reload_explicit_manipulator,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import GBManipulator, GBManipulatorError, ParentError

_MISSING = object()


def _normalize_candidate_id(candidate_id: object) -> str:
    """Validate one stable candidate identity.

    :param candidate_id: Candidate identity to validate.
    :return: Validated non-empty identity.
    :raises TypeError: If ``candidate_id`` is not a non-empty string.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise TypeError("candidate_id must be a non-empty string")
    return candidate_id


def _normalize_input_index(input_index: object) -> int:
    """Normalize one candidate population index.

    :param input_index: Candidate position to normalize.
    :return: Python integer candidate index.
    :raises TypeError: If ``input_index`` is Boolean or non-integral.
    """
    if isinstance(input_index, (bool, np.bool_)) or not isinstance(
        input_index, Integral
    ):
        raise TypeError("input_index must be a non-Boolean integer")
    return int(input_index)


def _normalize_objective(objective: object) -> float:
    """Normalize one finite candidate objective.

    :param objective: Candidate objective to normalize.
    :return: Finite Python float.
    :raises TypeError: If ``objective`` is Boolean or non-real.
    :raises ValueError: If ``objective`` is non-finite.
    """
    if isinstance(objective, (bool, np.bool_)) or not isinstance(objective, Real):
        raise TypeError("objective must be a non-Boolean real scalar")
    normalized = float(objective)
    if not np.isfinite(normalized):
        raise ValueError("objective must be finite")
    return normalized


def _validate_success(success: object) -> bool:
    """Validate one evaluation success flag.

    :param success: Candidate success flag.
    :return: Validated Python Boolean.
    :raises TypeError: If ``success`` is not exactly a Python ``bool``.
    """
    if type(success) is not bool:
        raise TypeError("success must be a bool")
    return success


def _validate_failure_reason(
    success: bool,
    failure_reason: object,
    *,
    context: str,
) -> None:
    """Validate failure context for one evaluation-like result.

    :param success: Whether the result succeeded.
    :param failure_reason: Failure diagnostic supplied with the result.
    :param context: Keyword argument, required. Result noun used in diagnostics.
    :raises ValueError: If failure context conflicts with ``success``.
    """
    if success:
        if failure_reason is not None:
            raise ValueError(f"successful {context} must not include a failure reason")
        return
    if not isinstance(failure_reason, str) or not failure_reason:
        raise ValueError(f"failed {context} requires a failure reason")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Aligned result for one explicit-ownership candidate evaluation.

    :param candidate_id: Stable logical candidate identity independent of artifact
        paths.
    :param input_index: Candidate position in the submitted population.
    :param objective: Normalized finite objective, or the optimizer-supplied penalty on
        failure.
    :param structure_path: Canonical evaluator artifact path, when available.
    :param mapping: Candidate-to-file ownership mapping, when available.
    :param manipulator: Validated reconstructed candidate, when successful.
    :param success: Whether evaluation and reconstruction both succeeded.
    :param failure_reason: Failure context when ``success`` is false.
    :raises TypeError: If scalar or path fields have invalid types.
    :raises ValueError: If objective is non-finite or success/failure fields are
        internally inconsistent.
    """

    candidate_id: str
    input_index: int
    objective: float
    structure_path: str | None
    mapping: CandidateFileMapping | None
    manipulator: GBManipulator | None
    success: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Normalize scalar fields and enforce coherent result state.

        :raises TypeError: If scalar or path fields have invalid types.
        :raises ValueError: If objective is non-finite or success/failure fields are
            internally inconsistent.
        """
        candidate_id = _normalize_candidate_id(self.candidate_id)
        input_index = _normalize_input_index(self.input_index)
        objective = _normalize_objective(self.objective)
        success = _validate_success(self.success)

        structure_path = self.structure_path
        if structure_path is not None:
            if not isinstance(structure_path, (str, Path)):
                raise TypeError("structure_path must be a path-like string or None")
            structure_path = str(structure_path)

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "input_index", input_index)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "structure_path", structure_path)

        if success:
            if (
                structure_path is None
                or self.mapping is None
                or self.manipulator is None
            ):
                raise ValueError(
                    "successful evaluation requires a structure path, mapping, and "
                    "manipulator"
                )
            _validate_failure_reason(success, self.failure_reason, context="evaluation")
            return

        _validate_failure_reason(success, self.failure_reason, context="evaluation")
        if self.manipulator is not None:
            raise ValueError("failed evaluation must not include a manipulator")


@dataclass(frozen=True, slots=True)
class CandidateEvaluationSummary:
    """Lightweight historical result that does not depend on evaluator artifacts.

    :param candidate_id: Stable logical candidate identity.
    :param input_index: Candidate position in the submitted population.
    :param objective: Normalized finite objective value or failure penalty.
    :param success: Whether evaluation and explicit reconstruction succeeded.
    :param failure_reason: Failure context when ``success`` is false.
    :raises TypeError: If identity, index, objective, or success has an invalid type.
    :raises ValueError: If objective is non-finite or failure context is inconsistent.
    """

    candidate_id: str
    input_index: int
    objective: float
    success: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate summary state independently of any filesystem artifact.

        :raises TypeError: If identity, index, objective, or success has an invalid
            type.
        :raises ValueError: If objective is non-finite or failure context is
            inconsistent.
        """
        candidate_id = _normalize_candidate_id(self.candidate_id)
        input_index = _normalize_input_index(self.input_index)
        objective = _normalize_objective(self.objective)
        success = _validate_success(self.success)
        _validate_failure_reason(success, self.failure_reason, context="summary")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "input_index", input_index)
        object.__setattr__(self, "objective", objective)

    @classmethod
    def from_evaluation(
        cls, record: CandidateEvaluation
    ) -> "CandidateEvaluationSummary":
        """Create a summary from one typed evaluation.

        :param record: Explicit-ownership evaluation to summarize.
        :return: Artifact-independent historical result.
        :raises TypeError: If ``record`` is not a ``CandidateEvaluation``.
        """
        if not isinstance(record, CandidateEvaluation):
            raise TypeError("record must be a CandidateEvaluation")
        # pyraisecontract: ignore=DOC115[ValueError] CandidateEvaluation has already
        #   enforced the same finite-objective and success/failure coherence required by
        #   CandidateEvaluationSummary.
        return cls(
            candidate_id=record.candidate_id,
            input_index=record.input_index,
            objective=record.objective,
            success=record.success,
            failure_reason=record.failure_reason,
        )

    def to_state(self) -> dict[str, object]:
        """Return deterministic JSON-safe summary state."""
        return {
            "candidate_id": self.candidate_id,
            "input_index": self.input_index,
            "objective": self.objective,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_state(cls, state: object) -> "CandidateEvaluationSummary":
        """Restore one summary from checkpoint state.

        :param state: JSON-decoded summary dictionary.
        :return: Validated artifact-independent summary.
        :raises TypeError: If state is not a dictionary or fields have invalid types.
        :raises ValueError: If state values are inconsistent.
        :raises KeyError: If required fields are absent.
        """
        if not isinstance(state, dict):
            raise TypeError("evaluation summary state must be a dictionary")
        return cls(
            candidate_id=state["candidate_id"],
            input_index=state["input_index"],
            objective=state["objective"],
            success=state["success"],
            failure_reason=state.get("failure_reason"),
        )


class ExplicitOwnershipEvaluator:
    """Adapt evaluator callbacks to explicit-ownership candidate results.

    :param GB: Reference grain-boundary construction.
    :param scalar_energy_func: Scalar evaluator callback.
    :param batch_energy_func: Optional ordered batch evaluator callback.
    :param local_random: Optimizer-owned random-number generator.
    :param allow_variable_cell: Allow validated orthogonal variable-cell relaxation.
    :param penalty: Objective value assigned to failed calculations.
    """

    def __init__(
        self,
        *,
        GB: GBMaker,
        scalar_energy_func: Callable,
        batch_energy_func: Callable | None,
        local_random: np.random.Generator,
        penalty: float,
        allow_variable_cell: bool = False,
    ) -> None:
        """Initialize the explicit-ownership evaluator adapter.

        :param GB: Keyword argument, required. Reference grain-boundary construction.
        :param scalar_energy_func: Keyword argument, required. Scalar evaluator
            callback.
        :param batch_energy_func: Keyword argument, required. Optional ordered batch
            callback; may be ``None``.
        :param local_random: Keyword argument, required. Optimizer-owned random-number
            generator.
        :param penalty: Keyword argument, required. Optimizer-owned objective value
            assigned to failed calculations.
        :param allow_variable_cell: Keyword argument, optional, defaults to ``False``.
            Allow validated orthogonal evaluator-returned box relaxation.
        :raises TypeError: If a callback, random generator, penalty, or Boolean option
            has an invalid type.
        :raises ValueError: If ``penalty`` is non-finite.
        """
        if not callable(scalar_energy_func):
            raise TypeError("scalar_energy_func must be callable")
        if batch_energy_func is not None and not callable(batch_energy_func):
            raise TypeError("batch_energy_func must be callable or None")
        if not isinstance(local_random, np.random.Generator):
            raise TypeError("local_random must be a numpy.random.Generator")
        if isinstance(penalty, (bool, np.bool_)) or not isinstance(penalty, Real):
            raise TypeError("penalty must be a non-Boolean real scalar")
        normalized_penalty = float(penalty)
        if not np.isfinite(normalized_penalty):
            raise ValueError("penalty must be finite")
        if not isinstance(allow_variable_cell, (bool, np.bool_)):
            raise TypeError("allow_variable_cell must be a Boolean")
        self.GB = GB
        self.scalar_energy_func = scalar_energy_func
        self.batch_energy_func = batch_energy_func
        self.local_random = local_random
        self.allow_variable_cell = bool(allow_variable_cell)
        self.penalty = normalized_penalty
        self._claimed_paths: set[Path] = set()

    def begin_run(self) -> None:
        """Reset run-local evaluator artifact identity tracking."""
        self._claimed_paths.clear()

    def claimed_paths_state(self) -> list[str]:
        """Return deterministic checkpoint state for claimed evaluator artifacts.

        :return: Canonical claimed paths in lexical order.
        """
        return sorted(str(path) for path in self._claimed_paths)

    def restore_claimed_paths(self, paths: object) -> None:
        """Restore run-local evaluator artifact identity from a checkpoint.

        Historical claimed artifacts need not still exist, but every entry must remain a
        canonical path-like string so later path-reuse checks are deterministic.

        :param paths: Serialized sequence of previously claimed artifact paths.
        :raises TypeError: If the state is not a sequence of path strings.
        """
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise TypeError("claimed evaluator paths must be a list of strings")
        self._claimed_paths = {Path(path).resolve() for path in paths}

    def _candidate_file_mapping(
        self,
        manipulator: GBManipulator,
        atoms: np.ndarray,
    ) -> CandidateFileMapping:
        """Build the transient file mapping for one candidate.

        :param manipulator: Candidate manipulator carrying persistent labels.
        :param atoms: Candidate atom rows aligned with those labels.
        :return: Validated candidate/file mapping.
        :raises GrainOwnershipError: If ownership did not propagate or geometry is
            inconsistent.
        """
        labels = manipulator.candidate_grain_labels
        if labels is None:
            raise GrainOwnershipError(
                "explicit-ownership mutation did not propagate grain labels"
            )
        parent = manipulator.parents[0]
        try:
            validate_formula_composition(atoms, parent.unit_cell)
        except CandidateAdmissibilityError as exc:
            raise GrainOwnershipError(
                f"candidate composition is inadmissible: {exc}"
            ) from exc
        return CandidateFileMapping.from_candidate(
            atoms,
            labels,
            box_dims=parent.box_dims,
            gb_plane_x=parent.gb_plane_x,
            inplane_periodic=parent.inplane_periodic,
            left_grain_x_bounds=parent.left_grain_x_bounds,
            right_grain_x_bounds=parent.right_grain_x_bounds,
            coordinate_tolerance=parent.coordinate_tolerance,
            normal_topology=parent.normal_topology,
        )

    def _failed_evaluation(
        self,
        candidate_id: str,
        input_index: int,
        reason: str,
        mapping: CandidateFileMapping | None = None,
        structure_path: str | None = None,
    ) -> CandidateEvaluation:
        """Create one penalty-bearing failed evaluation result.

        :param candidate_id: Stable logical candidate identity.
        :param input_index: Candidate position in the submitted population.
        :param reason: Human-readable failure context.
        :param mapping: Optional candidate/file mapping, defaults to ``None``; supplied
            when construction reached that stage.
        :param structure_path: Optional canonical artifact path, defaults to ``None``;
            supplied when available from the evaluator.
        :return: Failed evaluation carrying both penalty and failure context.
        :raises TypeError: If candidate identity, input index, or path state is invalid.
        :raises ValueError: If candidate identity, objective, or failure state is
            inconsistent.
        """
        return CandidateEvaluation(
            candidate_id=candidate_id,
            input_index=input_index,
            objective=self.penalty,
            structure_path=structure_path,
            mapping=mapping,
            manipulator=None,
            success=False,
            failure_reason=reason,
        )

    @staticmethod
    def _normalize_objective(objective: object) -> float:
        """Normalize one evaluator objective.

        :param objective: Evaluator-returned objective value.
        :return: Finite Python float.
        :raises TypeError: If the value is Boolean or non-real.
        :raises ValueError: If the value is non-finite.
        """
        return _normalize_objective(objective)

    @staticmethod
    def _diagnostic_path(structure_path: object) -> str | None:
        """Return a canonical diagnostic path when one was supplied.

        :param structure_path: Evaluator-returned path-like value.
        :return: Canonical path string, or None for a non-path value.
        :raises OSError: If path normalization fails at the filesystem layer.
        :raises RuntimeError: If path normalization encounters an unrecoverable path
            resolution error.
        :raises ValueError: If the path value is malformed.
        """
        diagnostic_path: str | None = None
        if isinstance(structure_path, (str, Path)):
            diagnostic_path = str(Path(structure_path).resolve())
        return diagnostic_path

    def _reload_mapping(
        self,
        structure_path: str,
        mapping: CandidateFileMapping,
    ) -> GBManipulator:
        """Validate and reconstruct one evaluator artifact.

        :param structure_path: Evaluator-returned structure path.
        :param mapping: Expected candidate ownership and geometry.
        :return: Reconstructed manipulator with the optimizer RNG attached.
        :raises FileNotFoundError: If the artifact does not exist.
        :raises LammpsDataError: If the artifact cannot be read unambiguously.
        :raises GrainOwnershipError: If the artifact changed candidate identity or
            cannot be reconstructed as a valid candidate.
        """
        try:
            manipulator = reload_explicit_manipulator(
                structure_path,
                candidate_mapping=mapping,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
                type_dict=self.GB.unit_cell.type_map,
                allow_variable_cell=self.allow_variable_cell,
            )
        except (ParentError, GBManipulatorError) as exc:
            raise GrainOwnershipError(
                "evaluator artifact could not reconstruct a valid candidate"
            ) from exc
        try:
            validate_formula_composition(
                manipulator.parents[0].whole_system,
                manipulator.parents[0].unit_cell,
            )
        except CandidateAdmissibilityError as exc:
            raise GrainOwnershipError(
                f"evaluator output composition is inadmissible: {exc}"
            ) from exc
        manipulator.rng = self.local_random
        return manipulator

    def _record_result(
        self,
        *,
        candidate_id: str,
        input_index: int,
        mapping: CandidateFileMapping,
        objective: object = _MISSING,
        structure_path: object = _MISSING,
    ) -> CandidateEvaluation:
        """Normalize and validate one callback result.

        Missing objective or structure output is treated as a failed calculation and
        receives the optimizer penalty. Differentiating structural failures from other
        evaluator failures is intentionally deferred until evaluators expose a typed
        failure classification.

        :param candidate_id: Keyword argument, required. Stable logical candidate
            identity.
        :param input_index: Keyword argument, required. Candidate position in the
            submitted population.
        :param mapping: Keyword argument, required. Candidate/file mapping established
            before evaluation.
        :param objective: Keyword argument, optional, defaults to ``_MISSING``.
            Evaluator-returned objective, or the internal missing sentinel.
        :param structure_path: Keyword argument, optional, defaults to ``_MISSING``.
            Evaluator-returned artifact path, or the internal missing sentinel.
        :return: Successful reconstructed evaluation or a penalty-bearing failure.
        :raises TypeError: If candidate identity, input index, or result scalar state is
            invalid.
        :raises ValueError: If candidate identity or result state is internally
            inconsistent.
        """
        missing_fields = []
        if objective is _MISSING or objective is None:
            missing_fields.append("objective")
        if structure_path is _MISSING or structure_path is None:
            missing_fields.append("final_dump")
        try:
            diagnostic_path = self._diagnostic_path(structure_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return self._failed_evaluation(
                candidate_id,
                input_index,
                f"invalid structure path: {type(exc).__name__}: {exc}",
                mapping,
            )
        if missing_fields:
            return self._failed_evaluation(
                candidate_id,
                input_index,
                "incomplete evaluator result missing " + ", ".join(missing_fields),
                mapping,
                diagnostic_path,
            )

        try:
            numeric_objective = self._normalize_objective(objective)
        except (TypeError, ValueError) as exc:
            return self._failed_evaluation(
                candidate_id,
                input_index,
                f"invalid objective: {exc}",
                mapping,
                diagnostic_path,
            )

        if diagnostic_path is None:
            return self._failed_evaluation(
                candidate_id,
                input_index,
                "evaluator did not return a structure path",
                mapping,
            )

        path = Path(diagnostic_path)
        if not path.is_file():
            return self._failed_evaluation(
                candidate_id,
                input_index,
                "evaluator did not return a valid structure path",
                mapping,
                diagnostic_path,
            )
        if path in self._claimed_paths:
            return self._failed_evaluation(
                candidate_id,
                input_index,
                (
                    "evaluator reused a structure path already assigned in this run: "
                    f"{path}"
                ),
                mapping,
                diagnostic_path,
            )

        try:
            manipulator = self._reload_mapping(diagnostic_path, mapping)
        except (
            OSError,
            LammpsDataError,
            GrainOwnershipError,
        ) as exc:
            return self._failed_evaluation(
                candidate_id,
                input_index,
                f"{type(exc).__name__}: {exc}",
                mapping,
                diagnostic_path,
            )

        self._claimed_paths.add(path)
        return CandidateEvaluation(
            candidate_id=candidate_id,
            input_index=input_index,
            objective=numeric_objective,
            structure_path=diagnostic_path,
            mapping=mapping,
            manipulator=manipulator,
            success=True,
        )

    @staticmethod
    def _checkpoint_metadata(record: CandidateEvaluation) -> dict:
        """Return JSON-safe typed state for one owned candidate result.

        :param record: Validated explicit-ownership evaluation.
        :return: Versioned success/failure metadata for a candidate sidecar.
        """
        return {
            "owned_evaluation_version": 1,
            "success": record.success,
            "failure_reason": record.failure_reason,
        }

    def _restore_checkpointed_result(
        self,
        checkpoint: CandidateCheckpoint,
        unique_id: str,
        input_index: int,
        mapping: CandidateFileMapping,
    ) -> CandidateEvaluation:
        """Reconstruct one completed owned evaluation from a candidate sidecar.

        Successful artifacts are revalidated through the authoritative reload path.
        Typed failures remain failures and are never reconsidered as parents. Legacy or
        batch-written raw results without owned metadata are normalized normally.

        :param checkpoint: Candidate checkpoint containing the completed result.
        :param unique_id: Candidate identifier in the checkpoint.
        :param input_index: Candidate position in population order.
        :param mapping: Fresh candidate-local ownership mapping.
        :return: Restored aligned candidate evaluation.
        :raises TypeError: If checkpointed candidate identity or scalar state has an
            invalid type.
        :raises ValueError: If checkpointed candidate state is internally inconsistent.
        :raises CheckpointError: If the recorded candidate result is unavailable.
        """
        objective, structure_path = checkpoint.get_result(unique_id)
        metadata = checkpoint.get_metadata(unique_id)
        if (
            metadata is not None
            and metadata.get("owned_evaluation_version") == 1
            and metadata.get("success") is False
        ):
            reason = metadata.get("failure_reason")
            if not isinstance(reason, str) or not reason:
                reason = "checkpointed explicit-ownership evaluation failed"
            return self._failed_evaluation(
                unique_id,
                input_index,
                reason,
                mapping,
                self._diagnostic_path(structure_path),
            )
        return self._record_result(
            candidate_id=unique_id,
            input_index=input_index,
            mapping=mapping,
            objective=objective,
            structure_path=structure_path,
        )

    @staticmethod
    def _checkpoint_record(
        checkpoint: CandidateCheckpoint | None,
        unique_id: str,
        record: CandidateEvaluation,
    ) -> None:
        """Persist one validated owned evaluation when checkpointing is enabled.

        :param checkpoint: Candidate sidecar, or ``None`` when checkpointing is
            disabled.
        :param unique_id: Candidate identifier in the checkpoint.
        :param record: Validated explicit-ownership evaluation.
        :raises CheckpointError: If the sidecar cannot be persisted.
        """
        if checkpoint is None:
            return
        checkpoint.record(
            unique_id,
            record.objective,
            record.structure_path,
            metadata=ExplicitOwnershipEvaluator._checkpoint_metadata(record),
        )

    def evaluate_candidate(
        self,
        manipulator: GBManipulator,
        atoms: np.ndarray,
        unique_id: str,
        input_index: int,
    ) -> CandidateEvaluation:
        """Evaluate and reconstruct one candidate.

        :param manipulator: Candidate manipulator carrying ownership state.
        :param atoms: Candidate atom rows.
        :param unique_id: Evaluator invocation identifier.
        :param input_index: Candidate position in the population.
        :return: Normalized candidate evaluation.
        :raises TypeError: If candidate identity, index, or normalized result state has
            an invalid type.
        :raises ValueError: If normalized candidate result state is internally
            inconsistent.
        """
        try:
            mapping = self._candidate_file_mapping(manipulator, atoms)
        except GrainOwnershipError as exc:
            return self._failed_evaluation(unique_id, input_index, str(exc))

        try:
            result = self.scalar_energy_func(
                self.GB,
                manipulator,
                atoms,
                unique_id,
            )
            objective, structure_path = result
        except Exception as exc:
            # The external evaluator callback is a deliberate recovery boundary.
            return self._failed_evaluation(
                unique_id,
                input_index,
                f"{type(exc).__name__}: {exc}",
                mapping,
            )

        return self._record_result(
            candidate_id=unique_id,
            input_index=input_index,
            mapping=mapping,
            objective=objective,
            structure_path=structure_path,
        )

    def evaluate_generation(
        self,
        population_manipulators: list[GBManipulator],
        population_structures: list[np.ndarray],
        population_lineages: list[list[str]],
        gen: int,
        unique_id: int | UUID,
        *,
        gen_checkpoint: CandidateCheckpoint | None = None,
        cached_evaluations: list[CandidateEvaluation | None] | None = None,
    ) -> list[CandidateEvaluation]:
        """Evaluate one index-aligned explicit-ownership population.

        :param population_manipulators: Candidate manipulators in population order.
        :param population_structures: Candidate atom arrays in population order.
        :param population_lineages: Candidate lineages in population order.
        :param gen: Generation index.
        :param unique_id: Run identifier used to construct callback IDs.
        :param gen_checkpoint: Keyword argument, optional, defaults to ``None``.
            Per-candidate recovery sidecar for this generation.
        :param cached_evaluations: Keyword argument, optional, defaults to ``None``.
            Successful evaluations aligned to unchanged carryover candidates. ``None``
            entries are evaluated normally.
        :return: One aligned typed evaluation per input candidate.
        :raises TypeError: If candidate identity, index, or normalized result state has
            an invalid type.
        :raises ValueError: If population arrays or batch results are not aligned, or if
            a batch result is not a dictionary.
        :raises RuntimeError: If an internal alignment invariant is lost.
        :raises CheckpointError: If candidate checkpoint state cannot be read or
            written.
        """
        population_length = len(population_structures)
        if not (
            len(population_manipulators)
            == len(population_lineages)
            == population_length
        ):
            raise ValueError(
                "explicit-ownership population manipulators, structures, and lineages "
                "must remain index-aligned"
            )
        if cached_evaluations is None:
            cached_evaluations = [None] * population_length
        elif len(cached_evaluations) != population_length:
            raise ValueError(
                "cached explicit-ownership evaluations must remain index-aligned"
            )

        records: list[CandidateEvaluation | None] = [None] * population_length
        for index, cached in enumerate(cached_evaluations):
            if cached is None:
                continue
            if (
                not cached.success
                or cached.structure_path is None
                or cached.mapping is None
                or cached.manipulator is None
            ):
                raise ValueError(
                    "only successful explicit-ownership evaluations may be reused"
                )
            if not Path(cached.structure_path).is_file():
                continue
            try:
                mapping = self._candidate_file_mapping(
                    population_manipulators[index],
                    population_structures[index],
                )
            except GrainOwnershipError:
                continue
            # pyraisecontract: ignore=DOC115[TypeError] Cached CandidateEvaluation
            #   instances were already type-validated; only the population index changes
            #   here, and enumerate always supplies an int.
            records[index] = CandidateEvaluation(
                candidate_id=cached.candidate_id,
                input_index=index,
                objective=cached.objective,
                structure_path=cached.structure_path,
                mapping=mapping,
                manipulator=population_manipulators[index],
                success=True,
            )

        unique_ids = [f"GA_{unique_id}_g{gen}_c{i}" for i in range(population_length)]
        if self.batch_energy_func is None:
            for index, (manipulator, atoms, candidate_id) in enumerate(
                zip(population_manipulators, population_structures, unique_ids)
            ):
                if records[index] is not None:
                    continue
                try:
                    mapping = self._candidate_file_mapping(manipulator, atoms)
                except GrainOwnershipError as exc:
                    record = self._failed_evaluation(candidate_id, index, str(exc))
                else:
                    if gen_checkpoint is not None and gen_checkpoint.is_done(
                        candidate_id
                    ):
                        record = self._restore_checkpointed_result(
                            gen_checkpoint,
                            candidate_id,
                            index,
                            mapping,
                        )
                    else:
                        record = self.evaluate_candidate(
                            manipulator,
                            atoms,
                            candidate_id,
                            index,
                        )
                self._checkpoint_record(gen_checkpoint, candidate_id, record)
                records[index] = record
            if any(record is None for record in records):
                raise RuntimeError(
                    "explicit-ownership evaluation lost candidate alignment"
                )
            return [record for record in records if record is not None]

        valid_indices: list[int] = []
        valid_mappings: list[CandidateFileMapping] = []
        for index, (manipulator, atoms) in enumerate(
            zip(population_manipulators, population_structures)
        ):
            if records[index] is not None:
                continue
            try:
                mapping = self._candidate_file_mapping(manipulator, atoms)
            except GrainOwnershipError as exc:
                records[index] = self._failed_evaluation(
                    unique_ids[index], index, str(exc)
                )
                continue
            valid_indices.append(index)
            valid_mappings.append(mapping)

        if valid_indices:
            pending_positions = [
                position
                for position, input_index in enumerate(valid_indices)
                if gen_checkpoint is None
                or not gen_checkpoint.is_done(unique_ids[input_index])
            ]
            pending_indices = [
                valid_indices[position] for position in pending_positions
            ]
            pending_mappings = [
                valid_mappings[position] for position in pending_positions
            ]
            try:
                if pending_indices:
                    raw_results = self.batch_energy_func(
                        self.GB,
                        [population_manipulators[index] for index in pending_indices],
                        [population_structures[index] for index in pending_indices],
                        [population_lineages[index] for index in pending_indices],
                        [unique_ids[index] for index in pending_indices],
                        checkpoint=gen_checkpoint,
                    )
                else:
                    raw_results = []
            except Exception as exc:
                # The external evaluator callback is a deliberate recovery boundary.
                for input_index, mapping in zip(pending_indices, pending_mappings):
                    record = self._failed_evaluation(
                        unique_ids[input_index],
                        input_index,
                        f"{type(exc).__name__}: {exc}",
                        mapping,
                    )
                    records[input_index] = record
                    self._checkpoint_record(
                        gen_checkpoint,
                        unique_ids[input_index],
                        record,
                    )
            else:
                if not isinstance(raw_results, list) or len(raw_results) != len(
                    pending_mappings
                ):
                    raise ValueError(
                        "explicit-ownership batch evaluation requires one ordered "
                        "result dictionary per input candidate"
                    )
                for result_index, result in enumerate(raw_results):
                    if not isinstance(result, dict):
                        raise TypeError(
                            f"explicit-ownership batch result {result_index} must be a "
                            "dictionary"
                        )
                for input_index, mapping, result in zip(
                    pending_indices,
                    pending_mappings,
                    raw_results,
                ):
                    record = self._record_result(
                        candidate_id=unique_ids[input_index],
                        input_index=input_index,
                        mapping=mapping,
                        objective=result.get("energy", _MISSING),
                        structure_path=result.get("final_dump", _MISSING),
                    )
                    records[input_index] = record
                    self._checkpoint_record(
                        gen_checkpoint,
                        unique_ids[input_index],
                        record,
                    )

            if gen_checkpoint is not None:
                for input_index, mapping in zip(valid_indices, valid_mappings):
                    if records[input_index] is not None:
                        continue
                    candidate_id = unique_ids[input_index]
                    record = self._restore_checkpointed_result(
                        gen_checkpoint,
                        candidate_id,
                        input_index,
                        mapping,
                    )
                    records[input_index] = record
                    self._checkpoint_record(gen_checkpoint, candidate_id, record)

        if any(record is None for record in records):
            raise RuntimeError("explicit-ownership evaluation lost candidate alignment")
        return [record for record in records if record is not None]
