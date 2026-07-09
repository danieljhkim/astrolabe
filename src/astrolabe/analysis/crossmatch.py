"""Positional cross-match between two catalogs.

Nearest-neighbour match on sky position using astropy's SkyCoord, keeping pairs within
a separation threshold. Pure function: two Tables in, one joined Table out.
"""

from __future__ import annotations

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack


def crossmatch(
    left: Table,
    right: Table,
    *,
    radius_arcsec: float = 1.0,
    ra_col: str = "ra",
    dec_col: str = "dec",
    suffix: str = "_2",
) -> Table:
    """Match each `left` row to its nearest `right` row within `radius_arcsec`.

    Returns a Table of matched pairs: all `left` columns plus all `right` columns
    (right columns that collide with left names get `suffix`), and a `sep_arcsec`
    column with the on-sky separation. Unmatched `left` rows are dropped.
    """
    if len(left) == 0 or len(right) == 0:
        out = left.copy()[:0]
        out["sep_arcsec"] = []  # type: ignore[assignment]
        return out

    # u.Quantity(col, u.deg) works whether the column already carries deg units
    # (converts) or is unitless (assigns), so callers needn't pre-tag units.
    left_coords = SkyCoord(
        ra=u.Quantity(left[ra_col], u.deg), dec=u.Quantity(left[dec_col], u.deg)
    )
    right_coords = SkyCoord(
        ra=u.Quantity(right[ra_col], u.deg), dec=u.Quantity(right[dec_col], u.deg)
    )

    idx, sep2d, _ = left_coords.match_to_catalog_sky(right_coords)
    keep = sep2d.arcsecond <= radius_arcsec

    matched_left = left[keep]
    matched_right = right[idx[keep]]

    # Disambiguate colliding column names before stacking.
    right_renamed = matched_right.copy()
    for col in right_renamed.colnames:
        if col in matched_left.colnames:
            right_renamed.rename_column(col, f"{col}{suffix}")

    out = hstack([matched_left, right_renamed], metadata_conflicts="silent")
    out["sep_arcsec"] = sep2d.arcsecond[keep]
    return out
