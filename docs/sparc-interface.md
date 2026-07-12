# SPARC exchange convention (ORB-10168)

The contract for the SPARC external-galaxy delivery consumed by the ws_orrery
universality-fit task (global-β vs per-galaxy-β scarcity fits — the fixed-length-scale
universality test of the scarcity rotation-curve form). Producer: tycho (this repo,
`scripts/deliver_sparc.py`). Consumer: faraday. Everything below is load-bearing for
the fit apparatus; the same information rides machine-readably in each dataset's
sidecar `semantics` field.

## Provenance

SPARC — *Spitzer Photometry and Accurate Rotation Curves*: Lelli, McGaugh &
Schombert 2016, AJ 152, 157. 175 late-type galaxies (spanning dwarfs through giant
spirals) with 3.6 μm surface photometry and HI/Hα rotation curves. Published
machine-readable tables, fetched by the `sparc` source adapter:

- `SPARC_Lelli2016c.mrt` (Table 1, per-galaxy) and `MassModels_Lelli2016c.mrt`
  (Table 2, per-radius) from `http://astroweb.cwru.edu/SPARC/`.

## Datasets

| dataset | kind | content |
|---|---|---|
| `sparc_galaxies` | catalog | per-galaxy properties, 175 rows |
| `sparc_rotcurves` | catalog | per-radius mass models, ~3400 rows |
| `sparc_rar` | derived | analysis subsample with g_obs/g_bar; lineage names both parents |

Parquet + JSON sidecar under `data/processed/<kind>/`, astrolabe house layout. Keys
join on `source_id` (the SPARC galaxy name, str).

## Column semantics

`sparc_galaxies` (per galaxy): `dist_mpc`/`dist_err_mpc` (distance, Mpc),
`incl_deg`/`incl_err_deg` (disk inclination, deg), `lum36_1e9lsun` (total 3.6 μm
luminosity, 10⁹ L☉), `r_eff_kpc`, `sb_eff_lsun_pc2`, **`r_disk_kpc`** (exponential
disk scale length at 3.6 μm, kpc — the length scale for the fixed-scale test),
`sb_disk_lsun_pc2`, `mhi_1e9msun`, `r_hi_kpc`, `v_flat_kms`/`v_flat_err_kms`
(0 where the flat part is not reached), `quality`, `hubble_type`, `dist_method`.

`sparc_rotcurves` (per radius): `r_kpc`, `v_obs_kms` ± `v_obs_err_kms`, and the
baryonic component curves `v_gas_kms`, `v_disk_kms`, `v_bul_kms`, plus local surface
brightnesses `sb_disk_lsun_pc2`, `sb_bul_lsun_pc2`.

`sparc_rar` adds per point: `g_obs_ms2` (= v_obs²/r), `g_obs_err_ms2`
(= 2·v_obs·e_v/r), `g_bar_ms2`, all in **m/s²**, with the host-galaxy `quality`,
`incl_deg`, `dist_mpc`, `r_disk_kpc`, `lum36_1e9lsun` joined on so the fit needs no
second table.

## Conventions (the ones that bite)

- **Mass-to-light.** `v_disk_kms` / `v_bul_kms` are the component curves at
  Υ[3.6] = 1. A component's contribution to g_bar scales linearly with Υ
  (velocity with √Υ). `sparc_rar` bakes in the standard Υ_disk = 0.5,
  Υ_bul = 0.7 (McGaugh, Lelli & Schombert 2016); the values used ride in the
  sidecar `query` and in `Table.meta["rar"]`. To refit Υ, rebuild g_bar from the
  component columns of `sparc_rotcurves` — don't rescale `g_bar_ms2`.
- **Negative component velocities.** `v_gas_kms` (rarely `v_disk_kms`) is negative
  where the enclosed density has a central depression. The convention everywhere is
  v·|v|/r — a signed contribution, not v². Ignoring the sign biases g_bar high in
  dwarf centers, exactly the regime the universality test cares about.
- **Quality flag.** `quality` is SPARC's Q: 1 high, 2 medium, 3 low (severe
  asymmetries / face-on). `sparc_rar` keeps Q ≤ 2 **and** incl ≥ 30° — the standard
  RAR sample cuts. Refit inclinations at your own risk; v_obs scales as 1/sin(i).
- **No sky coordinates.** SPARC publishes none; there are no `ra`/`dec` columns.
- **Units are in the column names** (`_kms`, `_kpc`, `_mpc`, `_ms2`, `_1e9lsun`,
  `_lsun_pc2`, `_deg`). No hidden factors of h — SPARC distances are as published.

## Reproducing / refreshing

`uv run python scripts/deliver_sparc.py` refetches both tables and rewrites all
three datasets (overwrite-by-name is a refresh; the sidecar `query` + `fetched_at`
record what and when).
