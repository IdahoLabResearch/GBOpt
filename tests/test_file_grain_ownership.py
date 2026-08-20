# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnership,
    GrainOwnershipError,
    LammpsDataError,
    read_lammps_data_file,
    read_lammps_dump_file,
    read_lammps_structure_file,
    reload_explicit_manipulator,
)
from GBOpt.GBManipulator import InterfaceCandidate
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.UnitCell import UnitCell


def _write_explicit_ownership_data(path: Path, rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Synthetic UO2 atoms\n\n")
        stream.write(f"{len(rows)} atoms\n")
        stream.write("2 atom types\n")
        stream.write("2.000000 10.000000 xlo xhi\n")
        stream.write("-1.000000 9.000000 ylo yhi\n")
        stream.write("5.000000 15.000000 zlo zhi\n")
        stream.write("\nAtoms\n\n")
        for atom_id, type_id, charge, x, y, z in rows:
            stream.write(
                f"{atom_id} {type_id} {charge:.6f} "
                f"{x:.6f} {y:.6f} {z:.6f}\n"
            )


def _write_named_lammps_data(
    path: Path,
    atoms: np.ndarray,
    box_dims: np.ndarray,
    *,
    ids: np.ndarray | None = None,
    declared_types: int | None = None,
    type_map: dict[str, int] | None = None,
) -> None:
    atom_ids = (
        np.arange(1, len(atoms) + 1, dtype=np.int64)
        if ids is None
        else np.asarray(ids)
    )
    species = tuple(dict.fromkeys(str(name) for name in atoms["name"]))
    if type_map is None:
        type_map = {name: index for index, name in enumerate(species, start=1)}
    if set(species) - set(type_map):
        raise AssertionError("type_map must define every emitted species")
    if declared_types is None:
        declared_types = len(type_map)

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Owned candidate\n\n")
        stream.write(f"{len(atoms)} atoms\n")
        stream.write(f"{declared_types} atom types\n")
        for axis, (lower, upper) in zip("xyz", box_dims, strict=True):
            stream.write(f"{lower:.12f} {upper:.12f} {axis}lo {axis}hi\n")

        stream.write("\nAtom Type Labels\n\n")
        for name, type_id in sorted(type_map.items(), key=lambda item: item[1]):
            if name in species:
                stream.write(f"{type_id} {name}\n")

        stream.write("\nAtoms\n\n")
        for atom_id, atom in zip(atom_ids, atoms, strict=True):
            stream.write(
                f"{int(atom_id)} {atom['name']} "
                f"{atom['x']:.12f} {atom['y']:.12f} {atom['z']:.12f}\n"
            )


def _write_two_frame_dump(
    path: Path,
    atoms: np.ndarray,
    box_dims: np.ndarray,
    *,
    first_header: str = "id typelabel x y z",
    first_bounds_header: str = "pp pp pp",
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for timestep, shift in ((0, 0.0), (1, 1.0)):
            stream.write("ITEM: TIMESTEP\n")
            stream.write(f"{timestep}\n")
            stream.write("ITEM: NUMBER OF ATOMS\n")
            stream.write(f"{len(atoms)}\n")
            bounds_header = first_bounds_header if timestep == 0 else "pp pp pp"
            stream.write(f"ITEM: BOX BOUNDS {bounds_header}\n")
            for lower, upper in box_dims:
                stream.write(f"{lower} {upper}\n")
            header = first_header if timestep == 0 else "id typelabel x y z"
            stream.write(f"ITEM: ATOMS {header}\n")
            for atom_id, atom in enumerate(atoms, start=1):
                if "id" not in header.split():
                    stream.write(
                        f"{atom['name']} {atom['x'] + shift} "
                        f"{atom['y']} {atom['z']}\n"
                    )
                else:
                    stream.write(
                        f"{atom_id} {atom['name']} {atom['x'] + shift} "
                        f"{atom['y']} {atom['z']}\n"
                    )


def _synthetic_ownership() -> GrainOwnership:
    return GrainOwnership(
        atom_ids=np.arange(1, 6, dtype=np.int64),
        labels=np.array(
            [
                LEFT_GRAIN_LABEL,
                RIGHT_GRAIN_LABEL,
                RIGHT_GRAIN_LABEL,
                RIGHT_GRAIN_LABEL,
                RIGHT_GRAIN_LABEL,
            ],
            dtype=np.int8,
        ),
        gb_plane_x=4.0,
        inplane_periodic=(True, False),
        right_grain_x_bounds=(4.0, 9.5),
        coordinate_tolerance=1.0e-8,
        periodic_outer_x_interface=False,
    )


def _owned_candidate() -> tuple[np.ndarray, np.ndarray, np.ndarray, UnitCell]:
    atoms = np.asarray(
        [
            ("Ni", 6.5, 1.0, 1.0),
            ("Ni", 3.0, 2.0, 2.0),
            ("Ni", 4.0, 3.0, 3.0),
            ("Ni", 4.5, 4.0, 4.0),
            ("Ni", 5.5, 5.0, 5.0),
            ("Ni", 6.0, 6.0, 6.0),
            ("Ni", 7.0, 7.0, 7.0),
            ("Ni", 8.0, 8.0, 8.0),
        ],
        dtype=Atom.atom_dtype,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    box_dims = np.asarray(
        [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        dtype=float,
    )
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fcc", 3.52, "Ni")
    return atoms, labels, box_dims, unit_cell


def _candidate_mapping(
    atoms: np.ndarray,
    labels: np.ndarray,
    box_dims: np.ndarray,
    *,
    normal_topology: BoundaryNormalTopology = (
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    ),
) -> CandidateFileMapping:
    return CandidateFileMapping.from_candidate(
        atoms,
        labels,
        box_dims=box_dims,
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        normal_topology=normal_topology,
    )


def _interface_candidate(
    topology: BoundaryNormalTopology,
) -> InterfaceCandidate:
    atoms = np.asarray(
        [
            ("U", 3.0, 0.0, 1.0),
            ("O", 7.5, 2.0, 3.0),
            ("O", 9.0, 4.0, 5.0),
            ("U", 14.0, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )
    return InterfaceCandidate(
        atoms=atoms,
        box_dims=np.asarray(
            [[2.0, 16.0], [-1.0, 9.0], [0.0, 10.0]],
            dtype=float,
        ),
        gb_plane_x=8.25,
        left_grain_x_bounds=(2.5, 8.0),
        right_grain_x_bounds=(8.5, 15.5),
        grain_labels=np.asarray([0, 0, 1, 1], dtype=np.int8),
        inplane_periodic=(True, True),
        normal_topology=topology,
        coordinate_tolerance=1.0e-10,
        interface_separation=0.5,
    )


def test_grain_ownership_is_defensive_and_rejects_bad_labels() -> None:
    ownership = _synthetic_ownership()
    labels = ownership.labels
    assert labels.flags.writeable is False
    with pytest.raises(ValueError):
        labels[0] = RIGHT_GRAIN_LABEL
    assert ownership.labels[0] == LEFT_GRAIN_LABEL

    with pytest.raises(GrainOwnershipError, match="grain labels"):
        GrainOwnership(
            atom_ids=np.array([1, 2]),
            labels=np.array([0, 2]),
            gb_plane_x=4.0,
            inplane_periodic=(True, True),
            right_grain_x_bounds=(4.0, 9.0),
            coordinate_tolerance=1.0e-8,
            periodic_outer_x_interface=True,
        )
    with pytest.raises(GrainOwnershipError, match="length"):
        GrainOwnership(
            atom_ids=np.array([1, 2]),
            labels=np.array([0]),
            gb_plane_x=4.0,
            inplane_periodic=(True, True),
            right_grain_x_bounds=(4.0, 9.0),
            coordinate_tolerance=1.0e-8,
            periodic_outer_x_interface=True,
        )


def test_grain_ownership_accepts_standard_real_scalars() -> None:
    ownership = GrainOwnership(
        atom_ids=np.asarray([1, 2], dtype=np.int64),
        labels=np.asarray([LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL], dtype=np.int8),
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )

    assert ownership.gb_plane_x == pytest.approx(5.0)
    assert ownership.coordinate_tolerance == pytest.approx(1.0e-8)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        pytest.param("gb_plane_x", True, id="boolean-plane"),
        pytest.param("gb_plane_x", "5.0", id="string-plane"),
        pytest.param("coordinate_tolerance", False, id="boolean-tolerance"),
        pytest.param(
            "coordinate_tolerance",
            "1e-8",
            id="string-tolerance",
        ),
    ],
)
def test_grain_ownership_rejects_coercive_real_scalars(
    field: str,
    invalid: object,
) -> None:
    arguments: dict[str, object] = {
        "atom_ids": np.asarray([1, 2], dtype=np.int64),
        "labels": np.asarray(
            [LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
            dtype=np.int8,
        ),
        "gb_plane_x": 5.0,
        "inplane_periodic": (True, True),
        "left_grain_x_bounds": (0.0, 5.0),
        "right_grain_x_bounds": (5.0, 10.0),
        "coordinate_tolerance": 1.0e-8,
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    }
    arguments[field] = invalid

    with pytest.raises(GrainOwnershipError, match=rf"{field} must be a real scalar"):
        GrainOwnership(**arguments)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        pytest.param(
            "left_grain_x_bounds",
            (0.0, True),
            id="boolean-left-bound",
        ),
        pytest.param(
            "right_grain_x_bounds",
            ("5.0", 10.0),
            id="string-right-bound",
        ),
    ],
)
def test_grain_ownership_rejects_coercive_x_bounds(field: str, invalid: object) -> None:
    arguments: dict[str, object] = {
        "atom_ids": np.asarray([1, 2], dtype=np.int64),
        "labels": np.asarray(
            [LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
            dtype=np.int8,
        ),
        "gb_plane_x": 5.0,
        "inplane_periodic": (True, True),
        "left_grain_x_bounds": (0.0, 5.0),
        "right_grain_x_bounds": (5.0, 10.0),
        "coordinate_tolerance": 1.0e-8,
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    }
    arguments[field] = invalid

    with pytest.raises(GrainOwnershipError, match=rf"{field}\[[01]\].*real scalar"):
        GrainOwnership(**arguments)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param((1, 0), id="integer-flags"),
        pytest.param(("yes", ""), id="truthy-string-flags"),
        pytest.param("pp", id="string-iterable"),
    ],
)
def test_grain_ownership_requires_explicit_boolean_periodicity(invalid: object) -> None:
    with pytest.raises(GrainOwnershipError, match="two Boolean flags"):
        GrainOwnership(
            atom_ids=np.asarray([1, 2], dtype=np.int64),
            labels=np.asarray(
                [LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
                dtype=np.int8,
            ),
            gb_plane_x=5.0,
            inplane_periodic=invalid,  # type: ignore[ty:invalid-argument-type]
            left_grain_x_bounds=(0.0, 5.0),
            right_grain_x_bounds=(5.0, 10.0),
            coordinate_tolerance=1.0e-8,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        )


def test_grain_ownership_translates_int64_id_overflow() -> None:
    with pytest.raises(GrainOwnershipError, match="signed 64-bit integer"):
        GrainOwnership(
            atom_ids=np.asarray([1, np.iinfo(np.int64).max + 1], dtype=object),
            labels=np.asarray(
                [LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
                dtype=np.int8,
            ),
            gb_plane_x=5.0,
            inplane_periodic=(True, True),
            left_grain_x_bounds=(0.0, 5.0),
            right_grain_x_bounds=(5.0, 10.0),
            coordinate_tolerance=1.0e-8,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        )


def test_grain_ownership_accepts_plane_inside_empty_central_interval() -> None:
    ownership = GrainOwnership(
        atom_ids=np.arange(1, 5),
        labels=np.array([0, 0, 1, 1]),
        gb_plane_x=7.25,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(2.0, 7.0),
        right_grain_x_bounds=(7.5, 12.5),
        coordinate_tolerance=1.0e-10,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )
    assert ownership.gb_plane_x == pytest.approx(7.25)
    assert ownership.left_grain_x_bounds is not None
    assert ownership.left_grain_x_bounds.tolist() == pytest.approx([2.0, 7.0])
    assert ownership.right_grain_x_bounds.tolist() == pytest.approx([7.5, 12.5])


def test_legacy_false_boolean_is_unknown_not_slab() -> None:
    ownership = GrainOwnership(
        atom_ids=np.arange(1, 3),
        labels=np.array([0, 1]),
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-10,
        periodic_outer_x_interface=False,
    )
    assert ownership.normal_topology is BoundaryNormalTopology.UNKNOWN
    assert not ownership.periodic_outer_x_interface


def test_ownership_aligns_labels_to_reordered_file_ids() -> None:
    ownership = _synthetic_ownership()
    reordered = ownership.aligned_to(np.asarray([5, 2, 1, 4, 3]))
    assert np.array_equal(reordered.atom_ids, np.asarray([5, 2, 1, 4, 3]))
    assert np.array_equal(reordered.labels, np.asarray([1, 1, 0, 1, 1]))


@pytest.mark.parametrize(
    "topology",
    [
        BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
    ],
)
def test_interface_candidate_conversions_preserve_geometry_and_topology(
    topology: BoundaryNormalTopology,
) -> None:
    candidate = _interface_candidate(topology)

    ownership = GrainOwnership.from_interface_candidate(candidate)
    assert ownership.normal_topology is topology
    assert np.array_equal(
        ownership.left_grain_x_bounds,
        candidate.left_grain_x_bounds,
    )
    assert np.array_equal(
        ownership.right_grain_x_bounds,
        candidate.right_grain_x_bounds,
    )

    mapping = CandidateFileMapping.from_interface_candidate(candidate)
    assert mapping.normal_topology is topology
    assert mapping.periodic_outer_x_interface == (
        topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    assert np.array_equal(mapping.box_dims, candidate.box_dims)
    assert np.array_equal(mapping.left_grain_x_bounds, candidate.left_grain_x_bounds)
    assert np.array_equal(mapping.right_grain_x_bounds, candidate.right_grain_x_bounds)

    reordered = mapping.ownership_for_file_ids(mapping.atom_ids[::-1])
    assert reordered.normal_topology is topology
    assert np.array_equal(reordered.atom_ids, mapping.atom_ids[::-1])
    assert np.array_equal(reordered.labels, mapping.labels[::-1])


def test_lammps_data_reader_preserves_file_row_ids_and_charge_coordinates(
    tmp_path: Path,
) -> None:
    data = tmp_path / "reader.data"
    rows = [
        (2, 2, -1.2, 3.0, 1.0, 7.0),
        (1, 1, 2.4, 7.0, 2.0, 8.0),
    ]
    _write_explicit_ownership_data(data, rows)

    parsed = read_lammps_data_file(data, type_dict={"U": 1, "O": 2})
    assert np.array_equal(parsed.atom_ids, np.array([2, 1]))
    assert parsed.atoms[0]["name"] == "O"
    assert parsed.atoms[0]["x"] == pytest.approx(3.0)
    assert parsed.atoms[1]["name"] == "U"
    assert parsed.atoms[1]["z"] == pytest.approx(8.0)

    returned_ids = parsed.atom_ids
    assert not returned_ids.flags.writeable
    with pytest.raises(ValueError):
        returned_ids[0] = 99
    assert np.array_equal(parsed.atom_ids, np.array([2, 1]))


def test_named_data_rows_allow_file_local_type_labels_to_differ_from_type_dict(
    tmp_path: Path,
) -> None:
    candidate = _interface_candidate(BoundaryNormalTopology.PERIODIC_BICRYSTAL)
    atoms = candidate.atoms
    box_dims = candidate.box_dims
    data = tmp_path / "named_local_types.data"
    _write_named_lammps_data(
        data,
        atoms,
        box_dims,
        type_map={"O": 1, "U": 2},
    )

    parsed = read_lammps_data_file(data, type_dict={"U": 1, "O": 2})

    assert np.array_equal(parsed.atoms["name"], atoms["name"])


def test_structure_reader_dispatches_data_and_dump_formats(tmp_path: Path) -> None:
    atoms, _labels, box_dims, _unit_cell = _owned_candidate()
    data = tmp_path / "candidate.data"
    dump = tmp_path / "candidate.dump"
    _write_named_lammps_data(data, atoms, box_dims)
    _write_two_frame_dump(dump, atoms, box_dims)

    parsed_data = read_lammps_structure_file(data, type_dict={"Ni": 1})
    parsed_dump = read_lammps_structure_file(dump, type_dict={"Ni": 1})

    assert parsed_data.selected_frame is None
    assert parsed_data.boundary_periodic is None
    assert parsed_dump.selected_frame == 0
    assert parsed_dump.boundary_periodic == (True, True, True)
    np.testing.assert_array_equal(parsed_data.atoms, atoms)
    np.testing.assert_array_equal(parsed_dump.atoms, atoms)


def test_candidate_mapping_reloads_reordered_rows_by_transient_id(
    tmp_path: Path,
) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    mapping = _candidate_mapping(atoms, labels, box_dims)
    order = np.asarray([4, 0, 7, 2, 1, 6, 5, 3])
    returned = tmp_path / "reordered.data"
    _write_named_lammps_data(
        returned,
        atoms[order],
        box_dims,
        ids=mapping.atom_ids[order],
    )

    reloaded = reload_explicit_manipulator(
        returned,
        candidate_mapping=mapping,
        unit_cell=unit_cell,
        gb_thickness=10.0,
        type_dict={"Ni": 1},
    )
    assert np.array_equal(reloaded.parents[0].grain_labels, labels)
    assert np.array_equal(reloaded.parents[0].initial_atom_ids, mapping.atom_ids)


@pytest.mark.parametrize(
    ("defect", "expected_error", "message"),
    [
        pytest.param(
            "missing",
            GrainOwnershipError,
            "atom count does not match",
            id="missing-atom",
        ),
        pytest.param(
            "duplicate",
            LammpsDataError,
            "atom IDs must be unique",
            id="duplicate-id",
        ),
        pytest.param(
            "renumber",
            GrainOwnershipError,
            "atom IDs do not match",
            id="renumbered-id",
        ),
        pytest.param(
            "species",
            GrainOwnershipError,
            "changed species/type",
            id="changed-species",
        ),
        pytest.param(
            "box",
            GrainOwnershipError,
            "changed box bounds",
            id="changed-box",
        ),
    ],
)
def test_candidate_reload_rejects_contract_violations(
    tmp_path: Path,
    defect: str,
    expected_error: type[Exception],
    message: str,
) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    mapping = _candidate_mapping(atoms, labels, box_dims)
    output_atoms = atoms.copy()
    output_ids = mapping.atom_ids.copy()
    output_box = box_dims.copy()
    declared_types = 1

    if defect == "missing":
        output_atoms = output_atoms[:-1]
        output_ids = output_ids[:-1]
    elif defect == "duplicate":
        output_ids[-1] = output_ids[0]
    elif defect == "renumber":
        output_ids[-1] = 999
    elif defect == "species":
        output_atoms[0]["name"] = "Cu"
        declared_types = 2
    elif defect == "box":
        output_box[0, 1] += 0.5

    returned = tmp_path / f"bad_{defect}.data"
    _write_named_lammps_data(
        returned,
        output_atoms,
        output_box,
        ids=output_ids,
        declared_types=declared_types,
        type_map={"Ni": 1, "Cu": 2},
    )

    with pytest.raises(expected_error, match=message):
        reload_explicit_manipulator(
            returned,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=10.0,
            type_dict={"Ni": 1, "Cu": 2},
        )


def test_candidate_reload_accepts_enabled_orthogonal_variable_cell(
    tmp_path: Path,
) -> None:
    atoms, labels, _box_dims, unit_cell = _owned_candidate()
    source_box = np.asarray(
        [[-2.0, 18.0], [-1.0, 9.0], [5.0, 15.0]],
        dtype=float,
    )
    target_box = np.asarray(
        [[-3.0, 21.0], [-2.0, 10.0], [4.0, 20.0]],
        dtype=float,
    )
    mapping = CandidateFileMapping.from_candidate(
        atoms,
        labels,
        box_dims=source_box,
        gb_plane_x=6.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(-1.5, 5.5),
        right_grain_x_bounds=(6.5, 17.0),
        coordinate_tolerance=1.0e-8,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )

    relaxed = atoms.copy()
    for axis_name, axis_index in zip("xyz", range(3), strict=True):
        source_lo, source_hi = source_box[axis_index]
        target_lo, target_hi = target_box[axis_index]
        reduced = (relaxed[axis_name] - source_lo) / (source_hi - source_lo)
        relaxed[axis_name] = target_lo + reduced * (target_hi - target_lo)

    order = np.arange(len(relaxed))[::-1]
    returned = tmp_path / "variable_cell.data"
    _write_named_lammps_data(
        returned,
        relaxed[order],
        target_box,
        ids=mapping.atom_ids[order],
    )

    reloaded = reload_explicit_manipulator(
        returned,
        candidate_mapping=mapping,
        unit_cell=unit_cell,
        gb_thickness=10.0,
        type_dict={"Ni": 1},
        allow_variable_cell=True,
    )
    parent = reloaded.parents[0]

    np.testing.assert_allclose(parent.box_dims, target_box)
    assert parent.gb_plane_x == pytest.approx(6.6)
    np.testing.assert_allclose(parent.left_grain_x_bounds, [-2.4, 6.0])
    np.testing.assert_allclose(parent.right_grain_x_bounds, [7.2, 19.8])
    np.testing.assert_array_equal(parent.grain_labels, labels)

    # Ownership remains persistent even when a left-owned atom lies to the right of
    # the affinely updated reference plane after relaxation.
    assert parent.grain_labels[0] == LEFT_GRAIN_LABEL
    assert parent.whole_system[0]["x"] > parent.gb_plane_x


def test_candidate_reload_variable_cell_flag_requires_boolean(tmp_path: Path) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    mapping = _candidate_mapping(atoms, labels, box_dims)
    returned = tmp_path / "owned.data"
    _write_named_lammps_data(returned, atoms, box_dims)

    with pytest.raises(TypeError, match="allow_variable_cell must be a Boolean"):
        reload_explicit_manipulator(
            returned,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=10.0,
            type_dict={"Ni": 1},
            allow_variable_cell=1,  # ty: ignore[invalid-argument-type]
        )


def test_explicit_reader_rejects_triclinic_data_box(tmp_path: Path) -> None:
    atoms, _labels, box_dims, _unit_cell = _owned_candidate()
    output = tmp_path / "triclinic.data"
    _write_named_lammps_data(output, atoms, box_dims)
    text = output.read_text(encoding="utf-8")
    output.write_text(
        text.replace("\nAtom Type Labels\n",
                     "\n0.1 0.0 0.0 xy xz yz\n\nAtom Type Labels\n"),
        encoding="utf-8",
    )

    with pytest.raises(LammpsDataError, match="orthogonal LAMMPS data boxes"):
        read_lammps_data_file(output)


def test_dump_reader_preserves_first_frame_semantics_without_concatenation(
    tmp_path: Path,
) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    mapping = _candidate_mapping(atoms, labels, box_dims)
    dump = tmp_path / "two_frames.dump"
    _write_two_frame_dump(dump, atoms, box_dims)

    parsed = read_lammps_dump_file(dump)
    assert parsed.selected_frame == 0
    assert len(parsed.atoms) == len(atoms)
    np.testing.assert_allclose(parsed.atoms["x"], atoms["x"])

    reloaded = reload_explicit_manipulator(
        dump,
        candidate_mapping=mapping,
        unit_cell=unit_cell,
        gb_thickness=10.0,
        type_dict={"Ni": 1},
    )
    assert np.array_equal(reloaded.parents[0].grain_labels, labels)


def test_malformed_selected_dump_frame_fails_even_when_later_frame_is_valid(
    tmp_path: Path,
) -> None:
    atoms, _labels, box_dims, _unit_cell = _owned_candidate()
    dump = tmp_path / "bad_first.dump"
    _write_two_frame_dump(dump, atoms, box_dims, first_header="typelabel x y z")

    with pytest.raises(LammpsDataError, match="missing atom attribute 'id'"):
        read_lammps_dump_file(dump)


def test_explicit_dump_requires_unambiguous_topology_flags(tmp_path: Path) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    mapping = _candidate_mapping(atoms, labels, box_dims)
    dump = tmp_path / "missing_topology.dump"
    _write_two_frame_dump(dump, atoms, box_dims, first_bounds_header="")

    with pytest.raises(GrainOwnershipError, match="unambiguous boundary topology"):
        reload_explicit_manipulator(
            dump,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=10.0,
            type_dict={"Ni": 1},
        )


def test_dump_reader_rejects_extra_rows_in_selected_frame(tmp_path: Path) -> None:
    atoms, _labels, box_dims, _unit_cell = _owned_candidate()
    dump = tmp_path / "extra_selected_row.dump"
    _write_two_frame_dump(dump, atoms, box_dims)
    text = dump.read_text(encoding="utf-8")
    marker = "ITEM: TIMESTEP\n1\n"
    extra = f"999 Ni 1.0 1.0 1.0\n{marker}"
    dump.write_text(text.replace(marker, extra, 1), encoding="utf-8")

    with pytest.raises(LammpsDataError, match="unexpected content"):
        read_lammps_dump_file(dump)


def test_candidate_reload_rejects_per_id_species_swap_with_same_aggregate_counts(
    tmp_path: Path,
) -> None:
    unit_cell = UnitCell()
    unit_cell.init_by_structure("rocksalt", 3.0, ("Na", "Cl"))
    atoms = np.asarray(
        [
            ("Na", 2.0, 1.0, 1.0),
            ("Cl", 3.0, 2.0, 2.0),
            ("Na", 6.0, 3.0, 3.0),
            ("Cl", 7.0, 4.0, 4.0),
        ],
        dtype=Atom.atom_dtype,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    box_dims = np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    mapping = _candidate_mapping(atoms, labels, box_dims)

    swapped = atoms.copy()
    swapped[0]["name"], swapped[1]["name"] = "Cl", "Na"
    output = tmp_path / "species_swap.data"
    _write_named_lammps_data(
        output,
        swapped,
        box_dims,
        declared_types=2,
        type_map={"Na": 1, "Cl": 2},
    )
    assert sorted(swapped["name"].tolist()) == sorted(atoms["name"].tolist())

    with pytest.raises(GrainOwnershipError, match="changed species"):
        reload_explicit_manipulator(
            output,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=6.0,
            type_dict={"Na": 1, "Cl": 2},
        )


def test_candidate_reload_rejects_changed_dump_topology(tmp_path: Path) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    mapping = _candidate_mapping(atoms, labels, box_dims)
    dump = tmp_path / "changed_topology.dump"

    with dump.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ITEM: TIMESTEP\n0\n")
        stream.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
        stream.write("ITEM: BOX BOUNDS ff pp pp\n")
        for lower, upper in box_dims:
            stream.write(f"{lower} {upper}\n")
        stream.write("ITEM: ATOMS id typelabel x y z\n")
        for atom_id, atom in enumerate(atoms, start=1):
            stream.write(
                f"{atom_id} {atom['name']} {atom['x']} "
                f"{atom['y']} {atom['z']}\n"
            )

    with pytest.raises(GrainOwnershipError, match="changed boundary topology"):
        reload_explicit_manipulator(
            dump,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=10.0,
            type_dict={"Ni": 1},
        )


@pytest.mark.parametrize(
    "topology",
    [
        BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
    ],
)
def test_interface_candidate_geometry_survives_write_and_explicit_reload(
    tmp_path: Path,
    topology: BoundaryNormalTopology,
) -> None:
    candidate = _interface_candidate(topology)
    mapping = CandidateFileMapping.from_interface_candidate(candidate)
    output = tmp_path / f"{topology.value}.data"
    order = np.arange(len(candidate.atoms))[::-1]
    _write_named_lammps_data(
        output,
        candidate.atoms[order],
        candidate.box_dims,
        ids=mapping.atom_ids[order],
        declared_types=2,
    )
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    reloaded = reload_explicit_manipulator(
        output,
        candidate_mapping=mapping,
        unit_cell=unit_cell,
        gb_thickness=4.0,
        type_dict={"U": 1, "O": 2},
    )
    parent = reloaded.parents[0]
    assert parent.normal_topology is topology
    assert parent.periodic_outer_x_interface == (
        topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    assert np.array_equal(parent.box_dims, candidate.box_dims)
    assert parent.gb_plane_x == candidate.gb_plane_x
    assert np.array_equal(parent.left_grain_x_bounds, candidate.left_grain_x_bounds)
    assert np.array_equal(parent.right_grain_x_bounds, candidate.right_grain_x_bounds)
    assert np.array_equal(parent.grain_labels, candidate.grain_labels)
    np.testing.assert_array_equal(parent.whole_system, candidate.atoms)
    np.testing.assert_array_equal(
        parent.left_grain,
        candidate.atoms[candidate.grain_labels == LEFT_GRAIN_LABEL],
    )
    np.testing.assert_array_equal(
        parent.right_grain,
        candidate.atoms[candidate.grain_labels == RIGHT_GRAIN_LABEL],
    )


def test_candidate_mapping_assigns_fresh_sequential_ids_for_current_rows() -> None:
    atoms, labels, box_dims, _unit_cell = _owned_candidate()
    reduced_atoms = atoms[[0, 2, 3, 5, 7]]
    reduced_labels = labels[[0, 2, 3, 5, 7]]

    mapping = _candidate_mapping(reduced_atoms, reduced_labels, box_dims)

    assert np.array_equal(
        mapping.atom_ids,
        np.arange(1, len(reduced_atoms) + 1, dtype=np.int64),
    )
    assert np.array_equal(mapping.labels, reduced_labels)
    assert np.array_equal(mapping.species, reduced_atoms["name"])


@pytest.mark.parametrize("file_format", ["data", "dump"])
def test_lammps_readers_translate_int64_atom_id_overflow(
    tmp_path: Path,
    file_format: str,
) -> None:
    atom_id = np.iinfo(np.int64).max + 1
    path = tmp_path / f"overflow.{file_format}"

    if file_format == "data":
        path.write_text(
            "Overflow ID\n\n"
            "1 atoms\n"
            "1 atom types\n"
            "0.0 10.0 xlo xhi\n"
            "0.0 10.0 ylo yhi\n"
            "0.0 10.0 zlo zhi\n"
            "\nAtoms\n\n"
            f"{atom_id} Ni 1.0 2.0 3.0\n",
            encoding="utf-8",
        )
        reader = read_lammps_data_file
    else:
        path.write_text(
            "ITEM: TIMESTEP\n"
            "0\n"
            "ITEM: NUMBER OF ATOMS\n"
            "1\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 10.0\n"
            "0.0 10.0\n"
            "0.0 10.0\n"
            "ITEM: ATOMS id typelabel x y z\n"
            f"{atom_id} Ni 1.0 2.0 3.0\n",
            encoding="utf-8",
        )
        reader = read_lammps_dump_file

    with pytest.raises(LammpsDataError, match="signed 64-bit integer"):
        reader(path)


def test_lammps_data_reader_rejects_ambiguous_reverse_type_mapping(
    tmp_path: Path,
) -> None:
    data = tmp_path / "ambiguous_mapping.data"
    _write_explicit_ownership_data(
        data,
        [(1, 1, 2.4, 3.0, 1.0, 7.0)],
    )

    with pytest.raises(LammpsDataError, match="mapped to both"):
        read_lammps_data_file(
            data,
            type_dict={"U": 1, "O": 1},
        )


@pytest.mark.parametrize("file_format", ["data", "dump"])
def test_lammps_readers_reject_non_symbol_species_labels_before_truncation(
    tmp_path: Path,
    file_format: str,
) -> None:
    path = tmp_path / f"long_label.{file_format}"

    if file_format == "data":
        path.write_text(
            "Long label\n\n"
            "1 atoms\n"
            "1 atom types\n"
            "0.0 10.0 xlo xhi\n"
            "0.0 10.0 ylo yhi\n"
            "0.0 10.0 zlo zhi\n"
            "\nAtoms\n\n"
            "1 Nickel 1.0 2.0 3.0\n",
            encoding="utf-8",
        )
        reader = read_lammps_data_file
    else:
        path.write_text(
            "ITEM: TIMESTEP\n"
            "0\n"
            "ITEM: NUMBER OF ATOMS\n"
            "1\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            "0.0 10.0\n"
            "0.0 10.0\n"
            "0.0 10.0\n"
            "ITEM: ATOMS id typelabel x y z\n"
            "1 Nickel 1.0 2.0 3.0\n",
            encoding="utf-8",
        )
        reader = read_lammps_dump_file

    with pytest.raises(LammpsDataError, match="unsupported atom species label"):
        reader(path)


def test_candidate_file_mapping_rejects_noncanonical_candidate_ids() -> None:
    atoms, labels, box_dims, _unit_cell = _owned_candidate()

    with pytest.raises(GrainOwnershipError, match="exactly 1..N"):
        CandidateFileMapping(
            atom_ids=np.arange(len(atoms), 0, -1, dtype=np.int64),
            labels=labels,
            species=atoms["name"],
            box_dims=box_dims,
            gb_plane_x=5.0,
            inplane_periodic=(True, True),
            left_grain_x_bounds=(0.0, 5.0),
            right_grain_x_bounds=(5.0, 10.0),
            coordinate_tolerance=1.0e-8,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        )
