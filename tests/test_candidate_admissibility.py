"""Exact formula-composition regressions for candidate admissibility."""

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.UnitCell import UnitCell
from GBOpt._candidate_admissibility import (
    CandidateAdmissibilityError,
    validate_formula_composition,
)


@pytest.fixture
def fluorite_uo2():
    cell = UnitCell()
    cell.init_by_structure("fluorite", 5.454, ("U", "O"))
    return cell


def _atoms(species):
    return np.asarray(
        [(name, float(index), 0.0, 0.0) for index, name in enumerate(species)],
        dtype=Atom.atom_dtype,
    )


def test_uo2_formula_composition_accepts_exact_formula_multiple(fluorite_uo2):
    formula_units = validate_formula_composition(
        _atoms(["U", "O", "O", "U", "O", "O"]),
        fluorite_uo2,
    )

    assert formula_units == 2


def test_formula_composition_supports_more_than_two_species():
    cell = UnitCell()
    species = ["H"] * 2 + ["He"] * 4 + ["Li"] * 6
    coordinates = np.column_stack(
        (
            np.linspace(0.0, 0.55, len(species)),
            np.zeros(len(species)),
            np.zeros(len(species)),
        )
    )
    cell.init_by_custom(
        coordinates,
        species,
        1.0,
        np.eye(3),
        np.eye(3),
        {},
        ratio={1: 2, 2: 4, 3: 6},
    )

    formula_units = validate_formula_composition(_atoms(species), cell)

    assert formula_units == 2


def test_uo2_formula_composition_rejects_nonintegral_formula_ratio(fluorite_uo2):
    with pytest.raises(
        CandidateAdmissibilityError,
        match="formula",
    ):
        validate_formula_composition(
            _atoms(["U", "U", "O", "O", "O"]),
            fluorite_uo2,
        )


def test_formula_composition_rejects_unexpected_species(fluorite_uo2):
    with pytest.raises(CandidateAdmissibilityError, match="species do not exactly"):
        validate_formula_composition(
            _atoms(["U", "O", "Xe"]),
            fluorite_uo2,
        )
