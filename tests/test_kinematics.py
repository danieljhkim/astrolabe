"""Kinematics: Galactocentric transform + rotation-curve binning (offline).

The fixture is built by inverse-transforming known circular orbits from the
Galactocentric frame to ICRS observables, so the tests exercise the real
astropy transform in both directions with a known answer and no network.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ICRS, CartesianDifferential, CartesianRepresentation, Galactocentric
from astropy.table import Table

from astrolabe.analysis import galactocentric_kinematics, rotation_curve

V_CIRC = 230.0  # km/s, flat input curve the tests must recover


def circular_orbit_table(
    n: int = 800,
    *,
    v_scatter_kms: float = 0.0,
    r_range_kpc: tuple[float, float] = (5.0, 12.0),
    seed: int = 42,
) -> Table:
    """Stars on prograde circular orbits at V_CIRC, expressed as ICRS observables."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(*r_range_kpc, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-0.3, 0.3, n)
    x, y = rho * np.cos(phi), rho * np.sin(phi)

    # Prograde tangential direction in astropy's Galactocentric frame is
    # (sin phi, -cos phi): it gives the Sun (at x=-R0, y=0) velocity +y.
    v = V_CIRC + rng.normal(0.0, v_scatter_kms, n)
    v_x, v_y = v * np.sin(phi), -v * np.cos(phi)

    gc = Galactocentric(
        CartesianRepresentation(x * u.kpc, y * u.kpc, z * u.kpc)
        .with_differentials(
            CartesianDifferential(
                v_x * u.km / u.s, v_y * u.km / u.s, np.zeros(n) * u.km / u.s
            )
        )
    )
    icrs = gc.transform_to(ICRS())
    t = Table()
    t["ra"] = icrs.ra.deg
    t["dec"] = icrs.dec.deg
    t["distance_pc"] = icrs.distance.to_value(u.pc)
    t["pmra"] = icrs.pm_ra_cosdec.to_value(u.mas / u.yr)
    t["pmdec"] = icrs.pm_dec.to_value(u.mas / u.yr)
    t["radial_velocity"] = icrs.radial_velocity.to_value(u.km / u.s)
    t.meta["true_rho_kpc"] = rho  # for position round-trip assertions
    return t


def test_galactocentric_kinematics_roundtrip():
    t = circular_orbit_table(n=200)
    out = galactocentric_kinematics(t)
    np.testing.assert_allclose(
        np.asarray(out["R_kpc"]), t.meta["true_rho_kpc"], rtol=1e-8
    )
    # Circular prograde orbits: v_phi = +V_CIRC, v_R = v_z = 0.
    np.testing.assert_allclose(np.asarray(out["v_phi_kms"]), V_CIRC, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["v_R_kms"]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["v_z_kms"]), 0.0, atol=1e-6)
    assert "galactocentric_params" in out.meta


def test_galactocentric_kinematics_missing_columns():
    with pytest.raises(ValueError, match="missing kinematic columns"):
        galactocentric_kinematics(Table({"ra": [1.0], "dec": [2.0]}))


def test_rotation_curve_recovers_flat_curve():
    t = circular_orbit_table(n=2000, v_scatter_kms=10.0)
    curve = rotation_curve(
        galactocentric_kinematics(t),
        r_min_kpc=5.0, r_max_kpc=12.0, bin_kpc=1.0, min_stars=30,
    )
    assert set(curve.colnames) == {"R_kpc", "v_c_kms", "v_c_err_kms", "n_stars"}
    assert len(curve) >= 5
    # Median per bin should sit on the input curve well within the scatter.
    np.testing.assert_allclose(np.asarray(curve["v_c_kms"]), V_CIRC, atol=3.0)
    assert (np.asarray(curve["v_c_err_kms"]) > 0).all()
    assert (np.asarray(curve["n_stars"]) >= 30).all()
    assert "method" in curve.meta


def test_rotation_curve_min_stars_drops_sparse_bins():
    t = circular_orbit_table(n=100, r_range_kpc=(5.0, 6.0))
    curve = rotation_curve(
        galactocentric_kinematics(t),
        r_min_kpc=5.0, r_max_kpc=25.0, bin_kpc=1.0, min_stars=30,
    )
    # All stars live in 5-6 kpc; no bin beyond should survive min_stars.
    assert (np.asarray(curve["R_kpc"]) < 6.5).all()


def test_rotation_curve_empty_selection():
    t = circular_orbit_table(n=50)
    curve = rotation_curve(
        galactocentric_kinematics(t),
        r_min_kpc=20.0, r_max_kpc=25.0, bin_kpc=1.0, min_stars=10,
    )
    assert len(curve) == 0
    assert set(curve.colnames) == {"R_kpc", "v_c_kms", "v_c_err_kms", "n_stars"}


def test_rotation_curve_requires_kinematics_columns():
    with pytest.raises(ValueError, match="galactocentric_kinematics"):
        rotation_curve(Table({"R_kpc": [8.0]}))
