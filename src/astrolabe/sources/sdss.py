"""SDSS adapter — via astroquery.sdss.

Proves the adapter seam (SPEC M4): SDSS returns objid/ra/dec, so normalization maps
`objid -> source_id` while ra/dec pass through. Same shape as Gaia — query modes:

    cone search : {"ra": deg, "dec": deg, "radius": deg, "limit": int?}
    SQL         : {"sql": "SELECT ... FROM PhotoObj ..."}  (SDSS CasJobs SQL)

The network call is isolated in `_run` for fixture-based testing.
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from .base import ensure_standard_columns


class SDSSSource:
    """Adapter over the SDSS SkyServer via astroquery."""

    name = "sdss"

    def query(self, params: dict[str, Any]) -> Table:
        if "sql" in params:
            table = self._run_sql(params["sql"])
        else:
            table = self._run_cone(params)
        return self._normalize(table)

    # -- network seams (monkeypatched in tests) ----------------------------
    def _run_cone(self, params: dict[str, Any]) -> Table:  # pragma: no cover - network
        from astroquery.sdss import SDSS

        try:
            ra = float(params["ra"])
            dec = float(params["dec"])
            radius = float(params["radius"])
        except KeyError as e:
            raise ValueError(
                "cone search needs ra, dec, radius (deg) — or pass sql= instead"
            ) from e
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        return SDSS.query_region(
            coord,
            radius=radius * u.deg,
            fields=["objid", "ra", "dec", "u", "g", "r", "i", "z"],
        )

    def _run_sql(self, sql: str) -> Table:  # pragma: no cover - network
        from astroquery.sdss import SDSS

        return SDSS.query_sql(sql)

    # -- normalization -----------------------------------------------------
    def _normalize(self, table: Table) -> Table:
        out = table.copy()
        for cand in ("objid", "objID", "OBJID"):
            if cand in out.colnames:
                out.rename_column(cand, "source_id")
                break
        return ensure_standard_columns(out, require=True)
