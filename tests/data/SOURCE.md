# Data Sources

This directory contains reference grain boundary data used by the GBOpt test
suite. The values are transcribed from the supplementary material of the two
publications below and are used to validate GBOpt's structure generation and
energy calculations against independently published results.

## `olmsted_2009_fcc_gb_energies.py`

Grain boundary geometry ($\Sigma$, misorientation index, P/Q matrices) and computed
energies for 388 fcc grain boundaries in Ni and Al.

> Olmsted, D. L., Foiles, S. M., & Holm, E. A. (2009). Survey of computed
> grain boundary properties in face-centered cubic metals: I. Grain boundary
> energy. *Acta Materialia*, 57(13), 3694–3703.
> https://doi.org/10.1016/j.actamat.2009.04.007

- Source: supplementary data published alongside the article.
- Used in GBOpt tests to: validate CSL/boundary construction and cross-check
  computed boundary energies against an independent reference set.

## `zhang_2022_uo2_ceo2_gb_energies.py`

Grain boundary geometry (misorientation, P/Q matrices) and computed energies
for a set of symmetric-tilt, asymmetric-tilt, twist, and mixed grain
boundaries in UO2 and CeO2, alongside reference energies from prior
interatomic potential studies (Basak for UO2, Gotte for CeO2).

> Zhang, Y., Hansen, E. D., Harbison, T., Masengale, S., French, J., &
> Aagesen, L. K. (2022). A molecular dynamics survey of grain boundary energy
> in uranium dioxide and cerium dioxide. *Journal of the American Ceramic
> Society*, 105, 4471–4486.
> https://doi.org/10.1111/jace.18340

- Source: supplementary data published alongside the article.
- Used in GBOpt tests to: validate boundary construction and energy
  calculations for fluorite-structure oxides, complementing the fcc-metal
  coverage from Olmsted et al. (2009) above.

## Notes

- Both datasets are also listed as `references` entries in the repository's
  top-level `CITATION.cff`. If you use GBOpt's test suite or these datasets
  in derived work, please cite the original articles above in addition to
  GBOpt itself.
- Values here are transcribed for testing purposes; consult the original
  supplementary material for the authoritative dataset.
