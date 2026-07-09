"""Shared fixtures — small in-memory Tables standing in for survey results (no network)."""

from __future__ import annotations

import astropy.units as u
import pytest
from astropy.table import Table


@pytest.fixture
def gaia_table() -> Table:
    """A tiny Gaia-shaped result: source_id/ra/dec + photometry + parallax."""
    t = Table()
    t["source_id"] = [1001, 1002, 1003, 1004]
    t["ra"] = [10.0, 10.001, 250.0, 250.5] * u.deg
    t["dec"] = [41.0, 41.0005, -20.0, -20.2] * u.deg
    t["phot_g_mean_mag"] = [15.2, 18.9, 12.1, 20.4] * u.mag
    t["bp_rp"] = [0.8, 1.9, 0.4, 2.5] * u.mag
    t["parallax"] = [5.0, 2.0, 10.0, -1.0] * u.mas
    return t


@pytest.fixture
def sdss_table() -> Table:
    """A tiny SDSS-shaped result: objid/ra/dec + ugriz (pre-normalization)."""
    t = Table()
    t["objid"] = [900001, 900002]
    t["ra"] = [10.0009, 250.4] * u.deg
    t["dec"] = [41.0004, -20.19] * u.deg
    t["g"] = [16.0, 19.0] * u.mag
    t["r"] = [15.5, 18.4] * u.mag
    return t
