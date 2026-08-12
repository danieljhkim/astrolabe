# Turnaround radii exchange convention (ORB-10754)

Contract for the measured turnaround / zero-velocity-surface compilation consumed
by kepler's principia study note (`cosmological-expansion-and-bound-systems.md`
and the shear-sourced-consumption evidence ledger). Producer: tycho (this repo,
`scripts/deliver_turnaround.py`). Consumer: kepler.

## Provenance

Literature compilation (offline curated table), not a survey fetch. Primary
observational program: Karachentsev and collaborators' Local Volume TRGB
distances + Hubble-flow R₀ measurements (2002–2018), plus cluster-scale R₀ for
Virgo (Karachentsev et al. 2014) and Fornax–Eridanus (Nasonova et al. 2011).
ΛCDM turnaround bound formula: Pavlidou & Tomaras 2014, JCAP 09, 020.

Every row carries a full citation, ADS bibcode, mass provenance strings, and a
boolean `circularity_flag` when `mass_analysis` was derived from the same R₀
measurement (the usual R₀ method). Prefer `mass_indep` + `ratio_meas_sc_indep`
for non-circular measured/predicted comparisons.

## Dataset

| dataset | kind | content |
|---|---|---|
| `turnaround_radii` | catalog | ~11 systems: R_ta ± err, masses, citations, prediction ratios |

Parquet + JSON sidecar under `data/processed/catalog/`. No `ra`/`dec` (system-level
compilation). Rebuild: `uv run python scripts/deliver_turnaround.py`.

## Prediction columns (stated cosmology in sidecar `query`)

Primary H₀ default **70 km s⁻¹ Mpc⁻¹**, Ω_m = 0.315, Ω_Λ = 0.685. Also tabulated
at H₀ = 67 and 73 for sensitivity.

| column | definition |
|---|---|
| `r_sc_mpc` | R_sc = (2 G M / H₀²)^{1/3} using `mass_analysis` — the shear-consumption scale |
| `r_pt_max_mpc` | Pavlidou–Tomaras R_ta,max = (G M / (Ω_Λ H₀²))^{1/3} |
| `r_lcdm_zv_mpc` | invert ΛCDM R₀–M relation M = (π²/8G) R₀³ H₀² / f(Ω_m)² |
| `ratio_meas_sc` | R_ta / R_sc (analysis mass; **circular when flag true**) |
| `ratio_meas_sc_indep` | R_ta / R_sc using independent mass |
| `ratio_meas_lcdm_zv` | R_ta / R_lcdm_zv (≈1 when mass is R₀-circular under same cosmology) |
| `ratio_meas_pt_max` | R_ta / R_pt,max (ΛCDM requires ≲ 1 for non-expanding shells) |

**H₀ sensitivity:** R_sc and R_pt scale as H₀^{−2/3}. Raising H₀ from 67 → 73
shrinks R_pred by (67/73)^{2/3} ≈ 0.945 (~5.5%). Ratio columns
`ratio_meas_sc_h67` / `ratio_meas_sc_h73` are on every row.

## Circularity discipline

Nearly all published R₀ papers convert R₀ → M with a spherical-collapse formula.
Comparing R_meas to R_pred(M_R₀) is then partly tautological for the ΛCDM R₀
inversion (`ratio_meas_lcdm_zv` ≈ 1 by construction) and still informative but
biased for R_sc. **Non-circular verdicts should use `mass_indep`.** Where the
literature is too heterogeneous (Sculptor filament, CVn cloud — long crossing
times), rows say so in `notes`; coverage honesty over volume.

## Source list (bibcodes)

- 2018A&A...609A..11K — Kashibadze & Karachentsev (LG R₀, synthetic stack)
- 2006Ap.....49....3K — Karachentsev & Kashibadze (LG + M81 classic R₀)
- 2007AJ....133..504K — Karachentsev et al. (Cen A)
- 2003A&A...408..111K — Karachentsev et al. (IC 342)
- 2003A&A...404...93K — Karachentsev et al. (Sculptor / NGC 253)
- 2003A&A...398..467K — Karachentsev et al. (CVn I / NGC 4736)
- 2002A&A...385...21K — Karachentsev et al. (Cen A / M83 distances)
- 2005AJ....129..178K — Karachentsev (Local Group and neighboring groups summary)
- 2014ApJ...782....4K — Karachentsev et al. (Virgo R₀)
- 2011A&A...532A.104N — Nasonova et al. (Fornax–Eridanus R₀)
- 2014JCAP...09..020P — Pavlidou & Tomaras (ΛCDM R_ta,max bound)
- 2014AJ....148...50K — Karachentsev & Kudrya (orbital masses)
- 2020A&A...635A.135K — Kashibadze et al. (Virgo virial mass)

## Reproducing

```bash
uv run python scripts/deliver_turnaround.py --data-dir data
uv run python scripts/deliver_turnaround.py --h0 67   # alternate primary H0
```
