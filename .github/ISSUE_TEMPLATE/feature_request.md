---
name: Feature request
about: Suggest a new capability, crystal structure, file format, optimizer strategy, or other enhancement
title: '[Feature] '
labels: enhancement
assignees: ''
---

<!-- Fill in the sections that apply. Delete any section that does not fit your request.
     Exception: the Acceptance criteria section is required — please do not delete it. -->

## What would you like GBOpt to do?

<!-- One or two sentences. Focus on the outcome, not the implementation. -->
<!-- Example: "GBOpt should be able to build bicrystals for hexagonal (HCP) crystal structures." -->

## Environment

<!-- Fill in all fields so maintainers can confirm whether the feature is already available. -->

- **GBOpt version:**
- **Python version:**
- **OS / platform:**

## Which part of the pipeline does this affect?

<!-- Check all that apply. -->

- [ ] **GBMaker** — bicrystal construction (misorientation, inclination, crystal structure, lattice parameter)
- [ ] **GBManipulator** — atom translations, insertions/deletions, soft-mode displacements, file I/O
- [ ] **GBMinimizer** — Monte Carlo or Genetic Algorithm optimizer, acceptance criteria, move sets
- [ ] **Calculator / file I/O** — energy calculator integration (`get_gbe` interface), LAMMPS file format, dump parsing, new output formats
- [ ] **Other / cross-cutting** — *(describe below)*

## Motivation

<!-- Is this request driven by a capability gap, a research need, or a usability friction?
     Pick the one that fits best and remove the others.
     If this is blocking active research, include a brief deadline or context. -->

**Capability gap:** *GBOpt cannot currently do X, which is needed for Y type of simulation.*

**Research need:** *This feature would unblock the following work:*

**Usability friction:** *The current workflow requires me to [workaround], which is error-prone / tedious because ...*

## Proposed behavior

<!-- Describe what the feature should do. A minimal pseudocode sketch or proposed function
     signature is more useful than prose alone. -->

    # Example — replace with your sketch
    GB = GBMaker(a0=3.52, structure="hcp", ...)
    GB.write_lammps("hcp_gb.dat")

## Acceptance criteria

<!-- Required — please do not delete this section.
     List specific, checkable conditions that must be true for this feature to be "done."
     This prevents scope creep and helps reviewers know when to close the issue.
     If you are unsure what "done" looks like, make your best guess — the maintainer will refine it. -->

- [ ] *e.g., `GBMaker` accepts `structure="hcp"` without raising an error*
- [ ] *e.g., output file passes a round-trip read/write check*
- [ ] *e.g., a regression test is added to `tests/test_gbmaker.py`*

## API / breaking-change impact

<!-- Would this change any existing method signatures, file formats, or user-facing behavior?
     If yes, describe what would break and how you propose to handle backward compatibility. -->

- [ ] No breaking changes expected
- [ ] Breaking change — *(describe affected APIs or file formats)*

## Alternatives considered

<!-- What have you already tried?
     First, describe any GBOpt APIs you explored and why they fell short
     (e.g., "I tried `GBManipulator.translate_right_grain` but it doesn't expose Y").
     Then, if applicable, note external tools you compared against
     (ASE, pyiron, atomsk, OVITO, VESTA, Babel) and why they don't fully solve the problem. -->

*e.g., I tried `GBManipulator.X` but it does not expose Y, which means I cannot ...*

*e.g., I used atomsk to generate the HCP structure and manually converted the output, but this breaks the GBManipulator workflow because ...*
