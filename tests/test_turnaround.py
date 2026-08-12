"""ORB-10754: turnaround literature compilation + prediction ratios (offline)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from astrolabe.analysis.turnaround import (
    annotate_predictions,
    f_omega_m,
    mass_from_r0_lcdm_msun,
    r_lcdm_zv_mpc,
    r_pt_max_mpc,
    r_sc_mpc,
)
from astrolabe.sources import SOURCE_NAMES, get_source
from astrolabe.sources.turnaround import COLUMN_SEMANTICS, TurnaroundSource
from astrolabe.store import Store


def test_registry_resolves_turnaround():
    assert "turnaround" in SOURCE_NAMES
    src = get_source("turnaround")
    assert src.name == "turnaround"
    assert src.kind == "catalog"


def test_literature_table_shape_and_citations():
    src = TurnaroundSource()
    t = src.query({})
    assert len(t) >= 8
    assert "source_id" in t.colnames
    assert "citation" in t.colnames
    assert "circularity_flag" in t.colnames
    assert "mass_analysis_provenance" in t.colnames
    # Every measured row must have a non-empty citation and bibcode.
    for row in t:
        assert str(row["citation"]).strip()
        assert str(row["bibcode"]).strip()
    # Circularity is recorded (most R0-method masses are circular).
    assert bool(np.any(t["circularity_flag"]))
    # Semantics cover measured columns.
    for col in (
        "r_ta_mpc",
        "mass_analysis_1e12msun",
        "circularity_flag",
        "citation",
    ):
        assert col in COLUMN_SEMANTICS


def test_f_omega_m_bounds():
    # f → 2/3 as Ωm → 1; f → 1 as Ωm → 0. Limits via values near the edges.
    assert f_omega_m(0.315) == pytest.approx(0.816, rel=0.05)
    assert 0.66 < f_omega_m(0.9) < 0.85
    assert 0.9 < f_omega_m(0.05) < 1.01
    with pytest.raises(ValueError):
        f_omega_m(1.0)


def test_r_sc_scaling():
    # R ∝ M^{1/3} and ∝ H0^{-2/3}
    r1 = r_sc_mpc(1e12, 70.0)
    r8 = r_sc_mpc(8e12, 70.0)
    assert r8 / r1 == pytest.approx(2.0, rel=1e-9)
    r67 = r_sc_mpc(1e12, 67.0)
    r73 = r_sc_mpc(1e12, 73.0)
    assert r67 / r73 == pytest.approx((73.0 / 67.0) ** (2.0 / 3.0), rel=1e-9)
    # Order-of-magnitude: 10^12 Msun at H0=70 → ~1.4 Mpc
    assert 1.2 < float(r1) < 1.6


def test_pavlidou_reference_scale():
    # Pavlidou & Tomaras: ~11.2 Mpc for 10^15 Msun at H0~67.3, ΩΛ=0.685
    r = float(r_pt_max_mpc(1e15, 67.3, omega_lambda=0.685))
    assert r == pytest.approx(11.2, rel=0.03)


def test_lcdm_r0_mass_roundtrip():
    r0 = 0.91
    m = float(mass_from_r0_lcdm_msun(r0, h0_kms=67.3, omega_m=0.315))
    # Kashibadze & Karachentsev 2018 quote M = 1.95e12 (R0)^3 at these params
    assert m / 1e12 == pytest.approx(1.95 * r0**3, rel=0.05)
    r_back = float(r_lcdm_zv_mpc(m, h0_kms=67.3, omega_m=0.315))
    assert r_back == pytest.approx(r0, rel=1e-6)


def test_annotate_predictions_ratios_and_circular_row():
    src = TurnaroundSource()
    measured = src.query({})
    out = annotate_predictions(measured, h0_kms=70.0)
    assert "ratio_meas_sc" in out.colnames
    assert "ratio_meas_lcdm_zv" in out.colnames
    assert "ratio_meas_sc_h67" in out.colnames
    assert "ratio_meas_sc_h73" in out.colnames
    assert "turnaround" in out.meta

    # Local_Group: circular mass → ratio_lcdm_zv near 1 when H0 matches inversion
    # (primary H0=70 here; mass was quoted at Planck H0=67.3 — allow offset)
    lg = out[out["source_id"] == "Local_Group"][0]
    assert bool(lg["circularity_flag"]) is True
    assert 0.4 < float(lg["ratio_meas_sc"]) < 1.2
    assert 0.5 < float(lg["ratio_meas_lcdm_zv"]) < 1.5
    # Independent mass larger → smaller ratio_meas_sc_indep than ratio_meas_sc
    assert float(lg["ratio_meas_sc_indep"]) < float(lg["ratio_meas_sc"])
    # H0 sensitivity: higher H0 → smaller R_sc → larger ratio
    assert float(lg["ratio_meas_sc_h73"]) > float(lg["ratio_meas_sc_h67"])


def test_annotate_missing_columns_raises():
    bad = Table({"source_id": ["x"]})
    with pytest.raises(ValueError, match="missing"):
        annotate_predictions(bad)


def test_store_roundtrip(tmp_path):
    src = TurnaroundSource()
    measured = src.query({})
    table = annotate_predictions(measured, h0_kms=70.0)
    store = Store(tmp_path)
    meta = store.write(
        table,
        name="turnaround_radii",
        source=src.name,
        kind=src.kind,
        query={"task": "ORB-10754"},
        semantics={**COLUMN_SEMANTICS},
    )
    assert meta.n_rows == len(table)
    back = store.read("turnaround_radii", kind="catalog")
    assert len(back) == len(table)
    assert set(table["source_id"]) == set(back["source_id"])
