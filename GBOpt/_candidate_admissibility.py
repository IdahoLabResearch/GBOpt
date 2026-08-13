"""Validate exact formula composition for in-memory interface candidates.

This internal interface-domain module owns composition invariants shared by
manipulation and evaluation. Potential-specific charge policy and optimizer fallback
policy do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np


class CandidateAdmissibilityError(ValueError):
    """Raised when candidate composition is outside the configured formula domain."""


@dataclass(frozen=True, slots=True)
class FormulaComposition:
    """Exact normalized species formula and candidate formula-unit count."""

    species_ratio: tuple[tuple[str, int], ...]
    formula_units: int


def _formula_ratio(unit_cell: Any) -> tuple[tuple[str, int], ...]:
    """Return the unit cell's type-indexed ratio as a normalized species formula.

    :param unit_cell: Unit cell providing ``ratio`` and ``type_map``.
    :return: Canonically ordered, greatest-common-divisor-normalized formula.
    :raises CandidateAdmissibilityError: If formula metadata are malformed.
    """
    try:
        raw_ratio = unit_cell.ratio
        type_map = unit_cell.type_map
    except AttributeError as exc:
        raise CandidateAdmissibilityError(
            "candidate formula validation requires unit-cell ratio and type mapping"
        ) from exc
    if not isinstance(raw_ratio, dict) or not raw_ratio:
        raise CandidateAdmissibilityError("unit-cell formula ratio must be nonempty")
    inverse = {int(type_id): str(species) for species, type_id in type_map.items()}
    values: list[tuple[str, int]] = []
    divisor = 0
    for type_id, coefficient in raw_ratio.items():
        if (
            isinstance(type_id, (bool, np.bool_))
            or isinstance(coefficient, (bool, np.bool_))
            or not isinstance(type_id, (int, np.integer))
            or not isinstance(coefficient, (int, np.integer))
            or int(coefficient) <= 0
        ):
            raise CandidateAdmissibilityError(
                "unit-cell formula ratio must contain positive integer coefficients"
            )
        try:
            species = inverse[int(type_id)]
        except KeyError as exc:
            raise CandidateAdmissibilityError(
                f"unit-cell formula type {int(type_id)} has no species mapping"
            ) from exc
        normalized = int(coefficient)
        divisor = gcd(divisor, normalized)
        values.append((species, normalized))
    return tuple(sorted((species, value // divisor) for species, value in values))


def validate_formula_composition(
    atoms: np.ndarray,
    unit_cell: Any,
) -> FormulaComposition:
    """Validate that candidate species counts are an exact formula multiple.

    :param atoms: Structured candidate atom rows containing a ``name`` field.
    :param unit_cell: Unit cell defining the exact type-indexed formula ratio.
    :return: Normalized formula and exact number of formula units.
    :raises CandidateAdmissibilityError: If atoms, species, or counts are invalid.
    """
    structured = np.asarray(atoms)
    if (
        structured.ndim != 1
        or structured.dtype.names is None
        or "name" not in structured.dtype.names
    ):
        raise CandidateAdmissibilityError(
            "candidate atoms must be a one-dimensional structured array with names"
        )
    ratio = _formula_ratio(unit_cell)
    names = np.asarray(structured["name"]).astype(str)
    expected_species = {species for species, _coefficient in ratio}
    actual_species = set(names.tolist())
    if actual_species != expected_species:
        raise CandidateAdmissibilityError(
            "candidate species do not exactly match the unit-cell formula: "
            f"expected {sorted(expected_species)}, observed {sorted(actual_species)}"
        )
    formula_units: int | None = None
    observed: list[str] = []
    for species, coefficient in ratio:
        count = int(np.count_nonzero(names == species))
        observed.append(f"{species}={count}")
        if count % coefficient:
            raise CandidateAdmissibilityError(
                "candidate composition is not an integer formula multiple: "
                + ", ".join(observed)
            )
        species_units = count // coefficient
        if formula_units is None:
            formula_units = species_units
        elif species_units != formula_units:
            raise CandidateAdmissibilityError(
                "candidate composition violates the unit-cell formula ratio: "
                + ", ".join(
                    f"{name}={int(np.count_nonzero(names == name))}"
                    for name, _value in ratio
                )
            )
    if formula_units is None or formula_units <= 0:
        raise CandidateAdmissibilityError(
            "candidate must contain at least one complete formula unit"
        )
    return FormulaComposition(ratio, formula_units)


def composition_delta_is_formula_multiple(
    first_counts: dict[str, int],
    second_counts: dict[str, int],
    species_ratio: tuple[tuple[str, int], ...],
) -> bool:
    """Return whether an exact species-count difference is a formula multiple.

    :param first_counts: First species-count vector.
    :param second_counts: Second species-count vector.
    :param species_ratio: Exact normalized formula vector.
    :return: Whether ``first_counts - second_counts`` is an integer formula multiple.
    """
    multiplier: int | None = None
    for species, coefficient in species_ratio:
        delta = first_counts[species] - second_counts[species]
        if delta % coefficient:
            return False
        species_multiplier = delta // coefficient
        if multiplier is None:
            multiplier = species_multiplier
        elif species_multiplier != multiplier:
            return False
    return True
