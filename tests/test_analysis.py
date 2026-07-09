"""M3: analysis — crossmatch + HR diagram, against fixture tables."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from astrolabe.analysis import crossmatch, hr_diagram
from astrolabe.analysis.hr import absolute_magnitude


def test_crossmatch_pairs_nearby(gaia_table, sdss_table):
    # sdss rows sit ~arcsec from two gaia rows; match within 5".
    matched = crossmatch(gaia_table, sdss_table, radius_arcsec=5.0)
    assert len(matched) == 2
    assert "sep_arcsec" in matched.colnames
    assert (matched["sep_arcsec"] <= 5.0).all()
    # colliding ra/dec from the right side get suffixed.
    assert "ra_2" in matched.colnames


def test_crossmatch_tight_radius_drops_all(gaia_table, sdss_table):
    matched = crossmatch(gaia_table, sdss_table, radius_arcsec=0.001)
    assert len(matched) == 0


def test_crossmatch_empty_input(gaia_table):
    empty = gaia_table[:0]
    matched = crossmatch(empty, gaia_table)
    assert len(matched) == 0


def test_absolute_magnitude_math():
    # parallax 10 mas -> distance 100 pc -> distance modulus 5 -> M = m - 5.
    m = np.array([15.0])
    par = np.array([10.0])
    np.testing.assert_allclose(absolute_magnitude(m, par), [10.0])


def test_absolute_magnitude_nonpositive_parallax_is_nan():
    out = absolute_magnitude(np.array([15.0, 15.0]), np.array([5.0, -1.0]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


def test_hr_diagram_returns_figure(gaia_table):
    fig = hr_diagram(gaia_table)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.yaxis_inverted()  # brighter = up


def test_hr_diagram_apparent_mode(gaia_table):
    fig = hr_diagram(gaia_table, absolute=False)
    assert "apparent" in fig.axes[0].get_ylabel()
