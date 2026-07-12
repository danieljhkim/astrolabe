"""Radial-acceleration-relation arrays over SPARC-shaped rotation curves.

One pure function: per rotation-curve point, the observed centripetal
acceleration and the acceleration the baryons alone would produce
(McGaugh, Lelli & Schombert 2016, PRL 117, 201101):

    g_obs = v_obs^2 / r
    g_bar = (v_gas*|v_gas| + Y_disk * v_disk*|v_disk| + Y_bul * v_bul*|v_bul|) / r

Component velocities are the SPARC mass models at mass-to-light ratio
Y[3.6] = 1; the standard assumptions are Y_disk = 0.5, Y_bul = 0.7. The v*|v|
form preserves SPARC's sign convention — a negative component velocity marks
a *negative* enclosed-density contribution (central gas depression), not an
imaginary speed. Accelerations are reported in m/s^2.

When a SPARC galaxies table is supplied, the standard analysis-sample cuts are
applied (quality flag Q <= 2, inclination >= 30 deg) and the per-galaxy
distance, disk scale length, and flags are joined onto the output so
downstream fits (e.g. the fixed-length-scale universality test) need no second
table. No network, no I/O side effects.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.table import Table

# (km/s)^2 / kpc in m/s^2 — the single unit conversion in the module.
_G_UNIT = ((u.km / u.s) ** 2 / u.kpc).to(u.m / u.s**2)

_ROTCURVE_REQUIRED = (
    "source_id", "r_kpc", "v_obs_kms", "v_obs_err_kms", "v_gas_kms", "v_disk_kms",
)
_GALAXY_REQUIRED = ("source_id", "quality", "incl_deg")
# Joined from the galaxies table when present — dist/scale-length ride along
# for fits keyed on a physical length scale.
_GALAXY_JOIN = ("quality", "incl_deg", "dist_mpc", "r_disk_kpc", "lum36_1e9lsun")

# Column semantics for delivery sidecars (ORB-10168).
COLUMN_SEMANTICS: dict[str, str] = {
    "source_id": "SPARC galaxy name (str)",
    "r_kpc": "galactocentric radius, kpc",
    "g_obs_ms2": "observed centripetal acceleration v_obs^2/r, m/s^2",
    "g_obs_err_ms2": "propagated error 2*v_obs*e_v_obs/r, m/s^2",
    "g_bar_ms2": "baryonic acceleration (gas + Y_disk*disk + Y_bul*bulge), m/s^2",
    "quality": "SPARC quality flag Q of the host galaxy (1 high, 2 medium)",
    "incl_deg": "host-galaxy disk inclination, deg",
    "dist_mpc": "host-galaxy distance, Mpc",
    "r_disk_kpc": "host-galaxy 3.6 um disk scale length, kpc",
    "lum36_1e9lsun": "host-galaxy total 3.6 um luminosity, 10^9 Lsun",
}


def radial_acceleration_relation(
    rotcurves: Table,
    galaxies: Table | None = None,
    *,
    ml_disk: float = 0.5,
    ml_bulge: float = 0.7,
    max_quality: int = 2,
    min_incl_deg: float = 30.0,
) -> Table:
    """Compute g_obs / g_bar per rotation-curve point.

    `rotcurves` is a SPARC-shaped mass-model table (see `sources.sparc`);
    `v_bul_kms` is treated as zero when absent. Points with non-positive
    radius or non-finite v_obs are dropped. With `galaxies` supplied, rows
    are cut to Q <= `max_quality` and inclination >= `min_incl_deg`, and the
    per-galaxy columns in the module docstring are joined on. Assumptions and
    cuts ride in the output metadata.
    """
    missing = [c for c in _ROTCURVE_REQUIRED if c not in rotcurves.colnames]
    if missing:
        raise ValueError(f"rotation-curve table is missing columns {missing}")

    out = rotcurves.copy()
    r = np.asarray(out["r_kpc"], dtype=float)
    v_obs = np.asarray(out["v_obs_kms"], dtype=float)
    keep = (r > 0) & np.isfinite(r) & np.isfinite(v_obs)

    if galaxies is not None:
        gal_missing = [c for c in _GALAXY_REQUIRED if c not in galaxies.colnames]
        if gal_missing:
            raise ValueError(f"galaxies table is missing columns {gal_missing}")
        by_name = {str(row["source_id"]): row for row in galaxies}
        passes: dict[str, bool] = {
            name: (int(row["quality"]) <= max_quality
                   and float(row["incl_deg"]) >= min_incl_deg)
            for name, row in by_name.items()
        }
        keep &= np.array(
            [passes.get(str(name), False) for name in out["source_id"]], dtype=bool
        )

    out = out[keep]
    r = np.asarray(out["r_kpc"], dtype=float)
    v_obs = np.asarray(out["v_obs_kms"], dtype=float)
    v_obs_err = np.asarray(out["v_obs_err_kms"], dtype=float)
    v_gas = np.asarray(out["v_gas_kms"], dtype=float)
    v_disk = np.asarray(out["v_disk_kms"], dtype=float)
    v_bul = (
        np.asarray(out["v_bul_kms"], dtype=float)
        if "v_bul_kms" in out.colnames
        else np.zeros_like(r)
    )

    out["g_obs_ms2"] = _G_UNIT * v_obs**2 / r
    out["g_obs_err_ms2"] = _G_UNIT * 2.0 * v_obs * v_obs_err / r
    out["g_bar_ms2"] = _G_UNIT * (
        v_gas * np.abs(v_gas)
        + ml_disk * v_disk * np.abs(v_disk)
        + ml_bulge * v_bul * np.abs(v_bul)
    ) / r

    if galaxies is not None:
        join_cols = [c for c in _GALAXY_JOIN if c in galaxies.colnames]
        for col in join_cols:
            out[col] = [by_name[str(name)][col] for name in out["source_id"]]

    out.meta["rar"] = {
        "ml_disk": ml_disk,
        "ml_bulge": ml_bulge,
        "formula": "g_bar = (v_gas|v_gas| + Y_disk v_disk|v_disk| "
                   "+ Y_bul v_bul|v_bul|)/r; g_obs = v_obs^2/r; m/s^2",
        "sample_cuts": (
            {"max_quality": max_quality, "min_incl_deg": min_incl_deg}
            if galaxies is not None
            else None
        ),
    }
    return out
