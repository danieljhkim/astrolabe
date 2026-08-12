"""Offline tests for wide-binary selection, ṽ(g_N), and Newtonian mocks."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from astrolabe.analysis.wide_binaries import (
    BASELINE_CUTS,
    absolute_g,
    binned_vtilde,
    chance_alignment_rate,
    make_synthetic_star_field,
    mass_from_abs_g,
    newtonian_mock_pairs,
    select_star_quality,
    select_wide_pairs,
    sensitivity_table,
)
from astrolabe.sources import get_source
from astrolabe.sources.wide_binaries import WideBinarySource


def test_mass_from_abs_g_solar_like():
    # M_G ~ 4.7 is roughly solar
    m = mass_from_abs_g(np.array([4.7]))
    assert 0.7 < float(m[0]) < 1.3


def test_absolute_g_known():
    # ϖ = 10 mas → 100 pc → DM = 5 → M = m - 5
    np.testing.assert_allclose(absolute_g(np.array([15.0]), np.array([10.0])), [10.0])


def test_newtonian_mock_covers_g_range():
    mock = newtonian_mock_pairs(3000, seed=1)
    assert len(mock) >= 1000
    g = np.asarray(mock["g_N_ms2"], dtype=float)
    assert np.nanmin(g) < 1e-10
    assert np.nanmax(g) > 1e-9
    vt = np.asarray(mock["vtilde"], dtype=float)
    assert np.isfinite(vt).sum() > 500
    # bound-ish: median ṽ should be O(1), not >> 10
    assert 0.1 < float(np.nanmedian(vt)) < 5.0


def test_select_wide_pairs_recovers_planted():
    stars = make_synthetic_star_field(n_pairs=80, n_field=100, seed=3)
    pairs = select_wide_pairs(stars, max_stars=None)
    assert len(pairs) >= 20
    assert "vtilde" in pairs.colnames
    assert "g_N_ms2" in pairs.colnames
    # planted pairs are Newtonian → median ṽ O(1)
    med = float(np.nanmedian(pairs["vtilde"]))
    assert 0.05 < med < 8.0


def test_binned_vtilde_with_mock():
    pairs = newtonian_mock_pairs(2000, seed=2)
    mock = newtonian_mock_pairs(2000, seed=99)
    bins = binned_vtilde(pairs, mock, min_pairs=5)
    assert len(bins) >= 2
    assert "vtilde_med" in bins.colnames
    assert "vtilde_mock_med" in bins.colnames
    # mock and data from same generative model → medians similar order
    ratio = np.asarray(bins["vtilde_med"]) / np.asarray(bins["vtilde_mock_med"])
    finite = np.isfinite(ratio)
    assert finite.any()
    assert np.median(ratio[finite]) < 5.0


def test_chance_alignment_shifted_field_not_huge():
    stars = make_synthetic_star_field(n_pairs=60, n_field=80, seed=5)
    rate, per_bin = chance_alignment_rate(stars, max_stars=None, ra_shift_deg=0.5)
    assert np.isfinite(rate)
    # planted bound pairs dominate → chance rate should be modest
    assert rate < 1.5
    assert "r_chance" in per_bin.colnames


def test_sensitivity_table_shape():
    stars = make_synthetic_star_field(n_pairs=50, n_field=40, seed=8)
    sens = sensitivity_table(stars, n_mock=500, max_stars=None, seed=8)
    assert len(sens) == 8  # 2 ecc × 2 ruwe × 2 regimes
    assert set(sens["ecc_prior"]) <= {"thermal", "flat"}


def test_select_star_quality_drops_bad():
    t = Table()
    t["source_id"] = [1, 2, 3]
    t["ra"] = [0.0, 0.1, 0.2]
    t["dec"] = [0.0, 0.0, 0.0]
    t["parallax"] = [10.0, 10.0, 1.0]  # third is beyond 200 pc
    t["parallax_error"] = [0.1, 0.1, 0.1]
    t["pmra"] = [1.0, 1.0, 1.0]
    t["pmdec"] = [1.0, 1.0, 1.0]
    t["phot_g_mean_mag"] = [12.0, 12.0, 12.0]
    t["ruwe"] = [1.0, 2.0, 1.0]  # second fails RUWE
    t["bp_rp"] = [1.0, 1.0, 1.0]
    out = select_star_quality(t)
    assert len(out) == 1


def test_wide_binaries_source_adql(monkeypatch):
    stars = make_synthetic_star_field(n_pairs=5, n_field=5, seed=1)
    captured = {}

    def fake_run(self, adql):
        captured["adql"] = adql
        return stars

    monkeypatch.setattr(WideBinarySource, "_run_adql", fake_run)
    src = get_source("wide_binaries")
    out = src.query({"mode": "local_volume", "limit": 1000})
    assert "parallax >" in captured["adql"]
    assert "gaiadr3.gaia_source" in captured["adql"]
    assert "CIRCLE" in captured["adql"]
    assert "source_id" in out.colnames
    assert src.name == "wide_binaries"
    assert src.kind == "catalog"


def test_wide_binaries_adql_passthrough(monkeypatch):
    stars = make_synthetic_star_field(n_pairs=2, n_field=2, seed=2)

    def fake_run(self, adql):
        return stars

    monkeypatch.setattr(WideBinarySource, "_run_adql", fake_run)
    out = WideBinarySource().query({"adql": "SELECT source_id, ra, dec FROM x"})
    assert len(out) == len(stars)


def test_pair_cuts_recorded_in_meta():
    stars = make_synthetic_star_field(n_pairs=30, n_field=20, seed=9)
    pairs = select_wide_pairs(stars, max_stars=None, ruwe_max=1.4)
    assert pairs.meta["pair_cuts"]["ruwe_max"] == 1.4
    assert pairs.meta["pair_cuts"]["f_triple_residual"] == BASELINE_CUTS["f_triple_residual"]


def test_missing_columns_raise():
    t = Table({"source_id": [1], "ra": [0.0], "dec": [0.0]})
    with pytest.raises(ValueError, match="missing columns"):
        select_wide_pairs(t)
