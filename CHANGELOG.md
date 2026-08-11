# Changelog

All notable changes to GBOpt are documented in this file.

## v0.2.0 — boundary-spec API and exact stoichiometric construction

### Added

- **BoundarySpec API** — `GBMaker.from_boundary_spec()` accepts structured spec objects
  (`PQSpec`, `CSLExactSpec`, `CSLApproxSpec`, `FiveDOFSpec`) as an alternative to the
  legacy positional-argument constructor.
- **Exact stoichiometric construction** — `mode="exact"` (default for all spec types)
  uses a period-aligned integer grain-filling pipeline that preserves crystallographic
  plane completeness and eliminates stoichiometry failures in multi-component GBs.
- **`inplane_periodic` coherence metadata** — `GBMaker` now exposes `inplane_periodic`
  as a public property indicating whether the interface plane is coherent in-plane.
- **First-class approximate mode** — `CSLApproxSpec` and `FiveDOFSpec(mode="approximate")`
  support incoherent interfaces without requiring the boundary to be re-expressed as a
  rational CSL.
- **`exactify_five_dof`** — converts any cubic-CSL boundary expressed in five-DOF
  notation to a `CSLExactSpec` at the nearest rational-CSL misorientation, allowing
  legacy five-DOF inputs to participate in exact stoichiometric construction.
- **`gb_params` CLI** — new `convert`, `describe`, `exactify`, and `canonicalize`
  subcommands.
- **Examples migration** — all `examples/` scripts updated from legacy `GBMaker(...)`
  calls to the appropriate spec type.
- **Supporting internals** — `BoundaryEmbedding` dataclass; `exact_csl.py` standalone
  CSL arithmetic module; `prefer_exact` construction mode.

### Fixed

- Fluorite Σ5 stoichiometry and general multi-component GB stoichiometry failures
  resolved by the exact integer filling path; `@pytest.mark.known_bug` removed from
  `test_fluorite_stoichiometric`.

### Deprecated

- The legacy `GBMaker(a0, structure, gb_thickness, misorientation, atom_types, ...)`
  positional constructor now issues a `DeprecationWarning`. Migrate to
  `GBMaker.from_boundary_spec(<spec>)`. The legacy path will be removed in a future
  release.

### Known limitations

- The exact construction path requires y- and z-box dimensions to be commensurate with
  the chosen repeat periods; boundaries with irrational in-plane periodicities must use
  `mode="approximate"`.
- `exactify_five_dof` is limited to rationalizable cubic-CSL inputs; non-cubic lattices
  and boundaries with no nearby rational misorientation are not yet supported.
- Approximate-angle snapping and oblique in-plane periodicity vectors are deferred to a
  future release.
