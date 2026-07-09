"""Hertzsprung–Russell diagram from Gaia photometry.

Colour–magnitude plot: BP-RP colour on x, absolute G magnitude on y (inverted, bright
at top). Absolute magnitude is derived from apparent G and parallax when available.
Returns a matplotlib Figure; never shows or saves it (caller decides).
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from matplotlib.figure import Figure


def absolute_magnitude(
    g_mag: np.ndarray, parallax_mas: np.ndarray
) -> np.ndarray:
    """Absolute magnitude from apparent mag + parallax (milliarcsec).

    Distance modulus: M = m + 5*log10(parallax_mas) - 10, valid for positive parallax.
    Non-positive parallaxes yield NaN (unusable distance).
    """
    parallax_mas = np.asarray(parallax_mas, dtype=float)
    g_mag = np.asarray(g_mag, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        abs_g = g_mag + 5.0 * np.log10(parallax_mas) - 10.0
    abs_g[parallax_mas <= 0] = np.nan
    return abs_g


def hr_diagram(
    table: Table,
    *,
    color_col: str = "bp_rp",
    mag_col: str = "phot_g_mean_mag",
    parallax_col: str = "parallax",
    absolute: bool = True,
) -> Figure:
    """Build an HR (colour–magnitude) diagram Figure from a photometry Table.

    With `absolute=True` and a parallax column present, plots absolute G magnitude;
    otherwise plots apparent G. The y-axis is inverted (brighter = up).
    """
    color = np.asarray(table[color_col], dtype=float)

    if absolute and parallax_col in table.colnames:
        mag = absolute_magnitude(table[mag_col], table[parallax_col])
        ylabel = "absolute G magnitude"
    else:
        mag = np.asarray(table[mag_col], dtype=float)
        ylabel = "apparent G magnitude"

    finite = np.isfinite(color) & np.isfinite(mag)

    fig = Figure(figsize=(6, 7))
    ax = fig.add_subplot(111)
    ax.scatter(color[finite], mag[finite], s=4, alpha=0.6, edgecolors="none")
    ax.set_xlabel(f"{color_col} (colour)")
    ax.set_ylabel(ylabel)
    ax.set_title("HR diagram")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    fig.tight_layout()
    return fig
