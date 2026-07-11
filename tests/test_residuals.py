"""Ephemeris-vs-baseline residuals — synthetic trajectories, no network (ORB-10076)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from astrolabe.analysis import ephemeris_residuals


def _trajectory(n: int = 12) -> Table:
    """A synthetic heliocentric trajectory on the exchange-convention grid."""
    t = Table()
    t["epoch_jd_tdb"] = 2457388.5 + 10.0 * np.arange(n)
    phase = np.linspace(0.0, 2.0 * np.pi, n)
    t["x_au"] = 1.5 * np.cos(phase)
    t["y_au"] = 1.5 * np.sin(phase)
    t["z_au"] = 0.02 * np.sin(phase)
    return t


def test_identical_trajectories_give_zero_residuals():
    eph = _trajectory()
    out = ephemeris_residuals(eph, eph.copy())
    assert np.allclose(out["dr_au"], 0.0)
    assert out.meta["rms_dr_au"] == 0.0
    assert out.meta["max_dr_au"] == 0.0
    assert out.meta["n_epochs"] == len(eph)


def test_known_offset_recovered_exactly():
    eph = _trajectory()
    base = eph.copy()
    base["x_au"] = base["x_au"] + 3e-6
    base["y_au"] = base["y_au"] + 4e-6
    out = ephemeris_residuals(eph, base)
    # 3-4-5 triangle: |dr| = 5e-6 AU at every epoch, sign = baseline - ephemeris.
    assert np.allclose(out["dx_au"], 3e-6)
    assert np.allclose(out["dy_au"], 4e-6)
    assert np.allclose(out["dz_au"], 0.0)
    assert np.allclose(out["dr_au"], 5e-6)
    assert out.meta["rms_dr_au"] == pytest.approx(5e-6)


def test_row_order_does_not_matter():
    eph = _trajectory()
    shuffled = eph.copy()[::-1]
    out = ephemeris_residuals(eph, shuffled)
    assert np.allclose(out["dr_au"], 0.0)
    assert np.all(np.diff(out["epoch_jd_tdb"]) > 0)  # sorted output


def test_epoch_grid_mismatch_raises():
    eph = _trajectory()
    base = eph.copy()
    base["epoch_jd_tdb"] = base["epoch_jd_tdb"] + 0.5  # half-day offset grid
    with pytest.raises(ValueError, match="no interpolation"):
        ephemeris_residuals(eph, base)


def test_row_count_mismatch_raises():
    eph = _trajectory()
    with pytest.raises(ValueError, match="epoch grids differ"):
        ephemeris_residuals(eph, eph[:-1].copy())


def test_missing_columns_raises():
    eph = _trajectory()
    bad = eph.copy()
    bad.remove_column("z_au")
    with pytest.raises(ValueError, match="baseline table is missing"):
        ephemeris_residuals(eph, bad)


def test_empty_tables_raise():
    empty = _trajectory()[:0]
    with pytest.raises(ValueError, match="empty"):
        ephemeris_residuals(empty, empty.copy())
