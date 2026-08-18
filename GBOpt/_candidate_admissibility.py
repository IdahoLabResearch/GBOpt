"""Validate exact formula composition for in-memory interface candidates.

This internal interface-domain module owns composition invariants shared by manipulation
and evaluation. Potential-specific charge policy and optimizer fallback policy do not
belong here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from GBOpt.UnitCell import UnitCellError


class CandidateAdmissibilityError(ValueError):
    """Raised when candidate composition is outside the configured formula domain."""


def _formula_ratio(unit_cell: Any) -> tuple[tuple[str, int], ...]:
    """Return authoritative formula metadata from a unit-cell-like object.

    :param unit_cell: Unit cell providing ``formula_ratio``.
    :return: Canonically ordered normalized species formula.
    :raises CandidateAdmissibilityError: If formula metadata are unavailable or
        malformed.
    """
    try:
        ratio = unit_cell.formula_ratio
    except (AttributeError, UnitCellError) as exc:
        raise CandidateAdmissibilityError(
            "candidate formula validation requires valid unit-cell formula metadata"
        ) from exc

    if not isinstance(ratio, tuple) or not ratio:
        raise CandidateAdmissibilityError(
            "unit-cell formula ratio must be a nonempty tuple"
        )

    normalized: list[tuple[str, int]] = []
    for item in ratio:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or isinstance(item[1], (bool, np.bool_))
            or not isinstance(item[1], (int, np.integer))
            or int(item[1]) <= 0
        ):
            raise CandidateAdmissibilityError(
                "unit-cell formula ratio must contain species/positive-integer pairs"
            )
        species, coefficient = item
        normalized.append((species, int(coefficient)))

    if len({species for species, _coefficient in normalized}) != len(normalized):
        raise CandidateAdmissibilityError(
            "unit-cell formula ratio must contain unique species"
        )
    return tuple(normalized)


def validate_formula_composition(
    atoms: np.ndarray,
    unit_cell: Any,
) -> int:
    """Validate that candidate species counts are an exact formula multiple.

    :param atoms: Structured candidate atom rows containing a ``name`` field.
    :param unit_cell: Unit cell defining the authoritative normalized formula ratio.
    :return: Exact number of formula units represented by ``atoms``.
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
            "candidate species do not exactly match the unit-cell formula: expected "
            f"{sorted(expected_species)}, observed {sorted(actual_species)}"
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
    return formula_units


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
