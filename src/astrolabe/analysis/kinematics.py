"""Galactocentric kinematics and the rotation curve.

Two pure functions over 6D Gaia-shaped Tables (ra, dec, distance, pmra, pmdec,
radial_velocity): transform observables into Galactocentric cylindrical
coordinates/velocities, then bin azimuthal velocity into a circular-velocity
curve v_c(R). No network, no I/O side effects.

Frame convention: astropy's `Galactocentric` frame with its documented v4.0
defaults (R0 = 8.122 kpc, solar motion from Drimmel & Poggio 2018 / GRAVITY);
the params used are recorded in the output table's metadata. Sign convention:
`v_phi` is reported positive in the direction of Galactic rotation.

Method honesty: `rotation_curve` takes a robust average of v_phi per radial
bin. That is NOT an asymmetric-drift-corrected circular velocity — for a mixed
stellar sample the mean v_phi lags true v_c by roughly 5–15 km/s (population
dependent). Callers doing dynamics against the output must either accept that
documented bias or correct it upstream by selecting a colder tracer population.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, Galactocentric
from astropy.table import Table

# Recorded in output metadata so datasets are self-describing about the frame.
_FRAME = Galactocentric()


def galactocentric_kinematics(
    table: Table,
    *,
    distance_col: str = "distance_pc",
) -> Table:
    """Add Galactocentric cylindrical positions/velocities to a 6D table.

    Expects columns `ra`, `dec` (deg), `pmra`, `pmdec` (mas/yr, pmra including
    cos(dec) per Gaia convention), `radial_velocity` (km/s), and a distance
    column (`distance_col`, parsecs). Returns a copy with added columns:
    `R_kpc`, `z_kpc` (cylindrical Galactocentric position) and `v_R_kms`,
    `v_phi_kms`, `v_z_kms` (cylindrical velocities, v_phi > 0 with rotation).
    """
    required = ("ra", "dec", "pmra", "pmdec", "radial_velocity", distance_col)
    missing = [c for c in required if c not in table.colnames]
    if missing:
        raise ValueError(f"table is missing kinematic columns {missing}")

    icrs = ICRS(
        ra=u.Quantity(table["ra"], u.deg),
        dec=u.Quantity(table["dec"], u.deg),
        distance=u.Quantity(table[distance_col], u.pc),
        pm_ra_cosdec=u.Quantity(table["pmra"], u.mas / u.yr),
        pm_dec=u.Quantity(table["pmdec"], u.mas / u.yr),
        radial_velocity=u.Quantity(table["radial_velocity"], u.km / u.s),
    )
    gc = icrs.transform_to(_FRAME)
    gc.representation_type = "cylindrical"
    gc.differential_type = "cylindrical"

    out = table.copy()
    out["R_kpc"] = gc.rho.to_value(u.kpc)
    out["z_kpc"] = gc.z.to_value(u.kpc)
    out["v_R_kms"] = gc.d_rho.to_value(u.km / u.s)
    # d_phi is an angular rate; v_phi = rho * d_phi. Astropy's phi increases
    # opposite to Galactic rotation, hence the sign flip to report v_phi > 0
    # for prograde stars.
    v_phi = -(gc.rho * gc.d_phi).to_value(u.km / u.s, equivalencies=u.dimensionless_angles())
    out["v_phi_kms"] = v_phi
    out["v_z_kms"] = gc.d_z.to_value(u.km / u.s)
    out.meta["galactocentric_params"] = {
        "galcen_distance_kpc": _FRAME.galcen_distance.to_value(u.kpc),
        "z_sun_pc": _FRAME.z_sun.to_value(u.pc),
        "galcen_v_sun_kms": list(_FRAME.galcen_v_sun.xyz.to_value(u.km / u.s)),
    }
    return out


def rotation_curve(
    table: Table,
    *,
    r_min_kpc: float = 4.0,
    r_max_kpc: float = 25.0,
    bin_kpc: float = 0.5,
    z_max_kpc: float = 0.5,
    min_stars: int = 50,
) -> Table:
    """Bin v_phi into a circular-velocity curve v_c(R) with uncertainties.

    Expects the columns `galactocentric_kinematics` adds. Selects the disk
    (|z| < `z_max_kpc`), then per radial bin reports the median v_phi as
    `v_c_kms` with `v_c_err_kms` = 1.4826*MAD/sqrt(n) (robust error on the
    median) and the star count `n_stars`. Bins with fewer than `min_stars`
    are dropped, not padded. See the module docstring for the asymmetric-drift
    caveat; the same note rides in the output metadata.
    """
    required = ("R_kpc", "z_kpc", "v_phi_kms")
    missing = [c for c in required if c not in table.colnames]
    if missing:
        raise ValueError(f"table is missing columns {missing}; "
                         "run galactocentric_kinematics first")

    r = np.asarray(table["R_kpc"], dtype=float)
    z = np.asarray(table["z_kpc"], dtype=float)
    v_phi = np.asarray(table["v_phi_kms"], dtype=float)
    disk = (np.abs(z) < z_max_kpc) & np.isfinite(r) & np.isfinite(v_phi)

    edges = np.arange(r_min_kpc, r_max_kpc + bin_kpc, bin_kpc)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = disk & (r >= lo) & (r < hi)
        n = int(sel.sum())
        if n < min_stars:
            continue
        v = v_phi[sel]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        rows.append((float((lo + hi) / 2), med, 1.4826 * mad / np.sqrt(n), n))

    out = Table(
        rows=rows or None,
        names=("R_kpc", "v_c_kms", "v_c_err_kms", "n_stars"),
        dtype=(float, float, float, int),
    )
    out.meta["method"] = (
        "median v_phi per R bin, |z| < "
        f"{z_max_kpc} kpc; NOT asymmetric-drift corrected (biased low ~5-15 km/s "
        "for a mixed sample); err = 1.4826*MAD/sqrt(n)"
    )
    if "galactocentric_params" in table.meta:
        out.meta["galactocentric_params"] = table.meta["galactocentric_params"]
    return out
