# Changelog

All notable changes to GBOpt are documented in this file.

## v0.2.0 — boundary specifications, complete exact construction, and interface-aware optimization

### Added

- **BoundarySpec API** — `GBMaker.from_boundary_spec()` accepts structured spec
  objects (`PQSpec`, `CSLExactSpec`, `CSLApproxSpec`, and `FiveDOFSpec`) as an
  alternative to the legacy positional constructor.
- **Complete exact decorated-site construction** — built-in unit cells provide
  immutable rational basis metadata used internally to enumerate and wrap every
  decorated site directly in repeated exact P/Q supercells.
- **Explicit construction modes** — `exact`, `prefer_exact`, and `approximate`
  distinguish exact coherent construction, warning-backed fallback, and
  intentionally approximate or incoherent interfaces.
- **Five-DOF exactification** — `exactify_five_dof` maps supported cubic-CSL
  five-DOF inputs to exact CSL specifications without requiring users to
  construct P/Q matrices manually.
- **Boundary topology metadata** — `BoundaryNormalTopology` distinguishes
  periodic bicrystals, single-interface slabs, and structures with unknown
  boundary-normal topology.
- **Composable interface candidates** — `InterfaceCandidate` carries atom rows,
  grain labels, the actual GB plane, physical grain bounds, box dimensions,
  periodicity, topology, coordinate tolerance, and applied separation.
- **Termination and translation controls** — `GBManipulator` supports explicit
  right-grain translation, periodic grain-local termination cycling, and
  finite-slab grain-local termination cycling.
- **Topology-aware interface separation** — periodic bicrystals and
  single-interface slabs apply normal separation with topology-specific box and
  vacuum behavior.
- **Explicit file-backed grain ownership** — validated ownership metadata
  preserves left/right labels and physical interface geometry across LAMMPS
  data and dump artifacts without treating LAMMPS IDs as permanent atom
  identities.
- **Ownership-aware GA handoff** — `GeneticAlgorithmMinimizer` can consume a
  file-backed initial structure with explicit ownership and preserve that state
  through scalar or batch evaluation, mutation, crossover, carryover, cloning,
  and artifact reload.
- **`inplane_periodic` coherence metadata** — `GBMaker` exposes machine-readable
  y/z periodicity information.
- **`gb_params` CLI** — adds `convert`, `describe`, `exactify`, and
  `canonicalize` subcommands.
- **Examples migration** — repository examples use boundary-spec inputs instead
  of the legacy positional `GBMaker(...)` constructor.
- **Supporting crystallography internals** — exact CSL, P/Q, quaternion, plane,
  reduction, embedding, and exactification operations are separated into
  focused crystallography modules.

### Fixed

- Exact fluorite and other decorated-basis constructions no longer discard
  valid sites near an x boundary by applying all-or-nothing Cartesian clipping
  to conventional-cell origins.
- Exact construction now preserves expected absolute decorated-site and species
  populations rather than validating only aggregate species ratios.
- Exact right-grain layers are no longer deleted merely to make the projected
  periodic gap at least as large as the projected central gap. Unequal
  nonnegative gaps are retained; true overlaps raise an error.
- Representative Zhang fluorite boundaries now retain their complete expected
  left-grain, right-grain, and whole-system populations.
- File-backed asymmetric bicrystals retain the actual GB plane and explicit
  left/right ownership rather than being repartitioned at the simulation-box
  midpoint.
- LAMMPS data and dump reloads validate row identity, species, population, box
  geometry, periodicity, and topology before reconstructing an optimizer
  candidate.
- Failed or incomplete explicit-ownership evaluations retain aligned failure
  records and receive the optimizer penalty without corrupting GA population
  ordering.
- Boundary-normal operations no longer infer slab physics from an ambiguous
  false legacy periodic-interface flag.

### Deprecated

- The legacy
  `GBMaker(a0, structure, gb_thickness, misorientation, atom_types, ...)`
  positional constructor issues a `DeprecationWarning`. Migrate to
  `GBMaker.from_boundary_spec(...)`. The legacy path will be removed in a future
  release.

### Known limitations

- Exact decorated-site construction requires validated rational basis metadata.
  Built-in GBOpt structures provide this metadata; arbitrary floating-point
  custom bases are not rationalized automatically.
- Exact supercell construction requires canonical right-handed,
  positive-determinant matrices. Negative-determinant supercells are rejected.
- The exact path requires the selected y/z repeat dimensions to be
  commensurate. Irrational in-plane periodicities require
  `mode="approximate"`.
- `exactify_five_dof` is limited to supported rationalizable cubic-CSL inputs.
  Non-cubic lattices and boundaries without a suitable nearby rational
  misorientation are not yet supported.
- Approximate-angle snapping and general oblique in-plane periodicity vectors
  remain deferred.
- Single-interface slab termination cycling moves each complete finite grain,
  coupling its GB-facing and free-surface terminations. Independent GB-only
  slab termination control is deferred.
- Explicit ownership-aware file handoff and reload are integrated with the
  genetic-algorithm path. Other optimizer paths require separate integration.
