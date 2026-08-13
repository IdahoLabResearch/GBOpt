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
    result = validate_formula_composition(
        _atoms(["U", "O", "O", "U", "O", "O"]),
        fluorite_uo2,
    )

    assert result.species_ratio == (("O", 2), ("U", 1))
    assert result.formula_units == 2


def test_uploaded_failure_composition_is_rejected(fluorite_uo2):
    atoms = np.empty(3272 + 6291, dtype=Atom.atom_dtype)
    atoms["name"][:3272] = "U"
    atoms["name"][3272:] = "O"
    atoms["x"] = 0.0
    atoms["y"] = 0.0
    atoms["z"] = 0.0

    with pytest.raises(
        CandidateAdmissibilityError,
        match="formula",
    ):
        validate_formula_composition(atoms, fluorite_uo2)


def test_formula_composition_rejects_unexpected_species(fluorite_uo2):
    with pytest.raises(CandidateAdmissibilityError, match="species do not exactly"):
        validate_formula_composition(
            _atoms(["U", "O", "Xe"]),
            fluorite_uo2,
        )
