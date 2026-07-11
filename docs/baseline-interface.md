# Solar-system baseline exchange convention (ORB-10076)

The contract for comparing astrolabe's JPL Horizons planetary ephemerides against the
pure-Newtonian n-body baseline from orrery `lab/sims/solar-system-nbody`. Producer of
the ephemeris side: tycho (this repo). Producer of the baseline side: kepler. The
residual is computed by `astrolabe.analysis.ephemeris_residuals` — it matches epochs
exactly and never interpolates, so everything below is load-bearing.

## Targets

Planetary-**system barycenters** (planet + moons), not planet centers — the right
comparison points for a point-mass n-body model. Baseline point masses should
therefore carry planetary-*system* GMs (e.g. Earth+Moon combined).

| slug | Horizons id | dataset |
|---|---|---|
| mercury | 1 | `ephemeris/mercury_2016_2026` |
| venus | 2 | `ephemeris/venus_2016_2026` |
| earth | 3 | `ephemeris/earth_2016_2026` |
| mars | 4 | `ephemeris/mars_2016_2026` |
| jupiter | 5 | `ephemeris/jupiter_2016_2026` |
| saturn | 6 | `ephemeris/saturn_2016_2026` |
| uranus | 7 | `ephemeris/uranus_2016_2026` |
| neptune | 8 | `ephemeris/neptune_2016_2026` |

## Epoch grid

- Timescale: **TDB**, expressed as Julian Date (`epoch_jd_tdb`).
- Grid: JD 2457388.5 + 10·k for k = 0…365 — i.e. 2016-01-01 00:00 TDB through
  2025-12-29 00:00 TDB (JD 2461038.5), 366 epochs at 10 d cadence. (The nominal
  2026-01-01 stop is not on the grid; Horizons emits full steps only.)
- The authoritative grid is the `epoch_jd_tdb` column of any of the ephemeris
  datasets. **The baseline must be evaluated at exactly these epochs** (fixed-step
  output or dense output sampled onto the grid); `ephemeris_residuals` raises if any
  epoch differs by more than 1e-6 d.

## Frame and origin

- Origin: **Sun body center** (Horizons center `500@10`) — heliocentric, *not* the
  solar-system barycenter.
- Axes: **ICRF** (Earth mean equator and equinox of J2000; astroquery
  `refplane="earth"`). *Not* the ecliptic frame — watch the ~23.4° obliquity if the
  sim's initial conditions were authored in ecliptic coordinates.
- A heliocentric frame is non-inertial. The sim should still integrate in an
  inertial frame with the Sun as a free body, then **output planet positions
  relative to the Sun's instantaneous position** (`r_planet − r_sun`, same for
  velocities). That difference *is* the heliocentric state and matches Horizons'
  `500@10` vectors apples-to-apples.

## Units

Positions in **AU**, velocities in **AU/day**. Conversions for an
AU / years / solar-masses integrator:

- 1 Julian year = 365.25 d, so `v[AU/d] = v[AU/yr] / 365.25`; a 10 d step is
  10/365.25 ≈ 0.027379 yr.
- GM☉ = k² with Gauss's constant k = 0.01720209895, i.e.
  GM☉ ≈ 2.9591220828e−4 AU³/d² (≈ 39.4769 AU³/yr² if working in years).
- Sim time t = 0 maps to JD 2457388.5 TDB. Initial conditions: take the
  first-epoch state (position *and* velocity) from each ephemeris dataset.

## File format

Parquet. One file per planet, or one combined file with a `target` column of slugs
from the table above. Required columns:

    target        : planet slug (str)          [required in combined files]
    epoch_jd_tdb  : Julian Date, TDB (float)
    x_au, y_au, z_au : heliocentric ICRF position, AU (float)

Optional but welcome: `vx_au_d, vy_au_d, vz_au_d`. Drop the export anywhere
readable (e.g. `orrery/lab/sims/solar-system-nbody/baseline/`); tycho ingests it
and writes the residual dataset as `derived/` with lineage naming both sides.

## Residuals

`astrolabe.analysis.ephemeris_residuals(ephemeris, baseline)` → per-epoch
`dx/dy/dz/dr_au` (baseline − ephemeris) with `rms_dr_au` / `max_dr_au` summary in
`meta`. Honest-results note: if residuals grow secularly from the first epoch or
show a ~23.4°-signature, suspect frame/timescale mismatch (this doc violated), not
dynamics — say so rather than tuning.
