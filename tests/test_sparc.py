"""ORB-10168: SPARC adapter normalization + radial-acceleration-relation arrays.

The adapter's network seam is monkeypatched with fixture Tables shaped like the
parsed MRT files; the RAR numbers are checked against astropy unit conversions
computed independently in the test.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from astrolabe.analysis.rar import radial_acceleration_relation
from astrolabe.sources import get_source
from astrolabe.sources.sparc import SPARCSource


def _raw_galaxies_table() -> Table:
    """SPARC Table 1 shape (pre-normalization), spanning dwarfs to giants."""
    t = Table()
    t["Galaxy"] = ["DDO154", "NGC2403", "NGC2841", "UGC02885", "F571-8"]
    t["T"] = [10, 6, 3, 5, 5]
    t["D"] = [4.04, 3.16, 14.1, 80.6, 53.3]
    t["e_D"] = [0.20, 0.16, 1.4, 8.1, 10.6]
    t["f_D"] = [2, 2, 2, 1, 1]
    t["Inc"] = [66.0, 63.0, 76.0, 64.0, 25.0]
    t["e_Inc"] = [3.0, 3.0, 2.0, 3.0, 5.0]
    t["L[3.6]"] = [0.053, 10.041, 93.01, 403.5, 19.0]
    t["e_L[3.6]"] = [0.003, 0.51, 4.7, 40.6, 2.0]
    t["Reff"] = [0.37, 1.39, 3.64, 11.4, 3.0]
    t["SBeff"] = [88.4, 1170.0, 2545.0, 731.0, 300.0]
    t["Rdisk"] = [0.37, 1.39, 3.64, 11.40, 2.98]
    t["SBdisk"] = [82.4, 1024.0, 3116.0, 833.0, 350.0]
    t["MHI"] = [0.275, 3.199, 10.66, 38.10, 4.0]
    t["RHI"] = [4.0, 19.5, 40.6, 71.4, 15.0]
    t["Vflat"] = [47.0, 131.2, 284.8, 289.5, 0.0]
    t["e_Vflat"] = [1.1, 2.6, 5.1, 7.3, 0.0]
    t["Q"] = [2, 1, 1, 3, 1]  # UGC02885 low quality; F571-8 low inclination
    t["Ref"] = ["a", "b", "c", "d", "e"]
    return t


def _raw_rotcurves_table() -> Table:
    """SPARC Table 2 shape (pre-normalization): a few radii per galaxy."""
    t = Table()
    t["ID"] = ["DDO154", "DDO154", "NGC2403", "NGC2841", "UGC02885", "F571-8"]
    t["D"] = [4.04, 4.04, 3.16, 14.1, 80.6, 53.3]
    t["R"] = [0.98, 2.94, 10.0, 20.0, 40.0, 8.0]
    t["Vobs"] = [23.0, 45.5, 100.0, 285.0, 290.0, 130.0]
    t["e_Vobs"] = [1.5, 2.0, 2.5, 5.0, 7.0, 4.0]
    t["Vgas"] = [10.9, -8.0, 40.0, 30.0, 60.0, 30.0]
    t["Vdisk"] = [12.6, 25.0, 50.0, 180.0, 200.0, 90.0]
    t["Vbul"] = [0.0, 0.0, 0.0, 150.0, 80.0, 0.0]
    t["SBdisk"] = [40.2, 10.0, 100.0, 500.0, 50.0, 60.0]
    t["SBbul"] = [0.0, 0.0, 0.0, 900.0, 100.0, 0.0]
    return t


@pytest.fixture
def galaxies(monkeypatch) -> Table:
    monkeypatch.setattr(SPARCSource, "_run", lambda self, which: _raw_galaxies_table())
    return SPARCSource().query({"table": "galaxies"})


@pytest.fixture
def rotcurves(monkeypatch) -> Table:
    monkeypatch.setattr(SPARCSource, "_run", lambda self, which: _raw_rotcurves_table())
    return SPARCSource().query({})


# -- adapter -----------------------------------------------------------------

def test_registry_resolves_sparc():
    src = get_source("sparc")
    assert src.name == "sparc"
    assert src.kind == "catalog"


def test_galaxies_normalized(monkeypatch):
    monkeypatch.setattr(SPARCSource, "_run", lambda self, which: _raw_galaxies_table())
    out = SPARCSource().query({"table": "galaxies"})
    for col in ("source_id", "dist_mpc", "incl_deg", "r_disk_kpc",
                "lum36_1e9lsun", "v_flat_kms", "quality"):
        assert col in out.colnames
    # SPARC publishes no sky coordinates; source_id is normalized to str.
    assert "ra" not in out.colnames and "dec" not in out.colnames
    assert list(out["source_id"])[:2] == ["DDO154", "NGC2403"]


def test_rotation_curves_default_and_normalized(monkeypatch):
    seen = {}

    def fake_run(self, which):
        seen["which"] = which
        return _raw_rotcurves_table()

    monkeypatch.setattr(SPARCSource, "_run", fake_run)
    out = SPARCSource().query({})
    assert seen["which"] == "rotation_curves"  # the default table
    for col in ("source_id", "r_kpc", "v_obs_kms", "v_obs_err_kms",
                "v_gas_kms", "v_disk_kms", "v_bul_kms"):
        assert col in out.colnames


def test_unknown_table_raises():
    with pytest.raises(ValueError, match="unknown SPARC table"):
        SPARCSource().query({"table": "photometry"})


def test_missing_required_column_raises(monkeypatch):
    bad = _raw_rotcurves_table()
    bad.remove_column("Vobs")
    monkeypatch.setattr(SPARCSource, "_run", lambda self, which: bad)
    with pytest.raises(ValueError, match="v_obs_kms"):
        SPARCSource().query({})


# -- radial-acceleration relation ---------------------------------------------

def _expected_g(v_kms: float, r_kpc: float) -> float:
    return float(((v_kms * u.km / u.s) ** 2 / (r_kpc * u.kpc)).to_value(u.m / u.s**2))


def test_rar_g_obs_numeric(rotcurves):
    out = radial_acceleration_relation(rotcurves)
    row = out[out["source_id"] == "NGC2403"][0]
    assert np.isclose(row["g_obs_ms2"], _expected_g(100.0, 10.0), rtol=1e-9)
    assert np.isclose(
        row["g_obs_err_ms2"], 2 * 100.0 * 2.5 / 100.0**2 * _expected_g(100.0, 10.0),
        rtol=1e-9,
    )


def test_rar_g_bar_ml_scaling_and_gas_sign(rotcurves):
    out = radial_acceleration_relation(rotcurves, ml_disk=0.5, ml_bulge=0.7)
    # NGC2403 at R=10: gas 40, disk 50, no bulge.
    row = out[out["source_id"] == "NGC2403"][0]
    expected = (40.0**2 + 0.5 * 50.0**2) / 100.0**2 * _expected_g(100.0, 10.0)
    assert np.isclose(row["g_bar_ms2"], expected, rtol=1e-9)
    # DDO154 at R=2.94 has Vgas = -8: the v*|v| convention subtracts it.
    row = out[(out["source_id"] == "DDO154") & (out["r_kpc"] > 2)][0]
    expected = (-(8.0**2) + 0.5 * 25.0**2) / (45.5**2) * _expected_g(45.5, 2.94)
    assert np.isclose(row["g_bar_ms2"], expected, rtol=1e-9)
    # Bulge term scales with ml_bulge (NGC2841 has one).
    heavier = radial_acceleration_relation(rotcurves, ml_disk=0.5, ml_bulge=1.4)
    g0 = out[out["source_id"] == "NGC2841"][0]["g_bar_ms2"]
    g1 = heavier[heavier["source_id"] == "NGC2841"][0]["g_bar_ms2"]
    assert g1 > g0


def test_rar_quality_and_inclination_cuts(rotcurves, galaxies):
    out = radial_acceleration_relation(rotcurves, galaxies)
    names = set(out["source_id"])
    assert "UGC02885" not in names  # Q = 3
    assert "F571-8" not in names  # incl 25 < 30 deg
    assert {"DDO154", "NGC2403", "NGC2841"} == names
    # Per-galaxy columns joined for the fixed-length-scale fit.
    for col in ("quality", "incl_deg", "dist_mpc", "r_disk_kpc", "lum36_1e9lsun"):
        assert col in out.colnames
    row = out[out["source_id"] == "DDO154"][0]
    assert row["r_disk_kpc"] == pytest.approx(0.37)
    assert out.meta["rar"]["sample_cuts"] == {"max_quality": 2, "min_incl_deg": 30.0}


def test_rar_missing_columns_raise(rotcurves, galaxies):
    bad = rotcurves.copy()
    bad.remove_column("v_gas_kms")
    with pytest.raises(ValueError, match="v_gas_kms"):
        radial_acceleration_relation(bad)
    badgal = galaxies.copy()
    badgal.remove_column("quality")
    with pytest.raises(ValueError, match="quality"):
        radial_acceleration_relation(rotcurves, badgal)


def test_rar_meta_records_assumptions(rotcurves):
    out = radial_acceleration_relation(rotcurves, ml_disk=0.4)
    assert out.meta["rar"]["ml_disk"] == 0.4
    assert out.meta["rar"]["sample_cuts"] is None  # no galaxies table given
