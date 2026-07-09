"""Gaia adapter — via astroquery.gaia (TAP/ADQL).

Two query modes:

    cone search : {"ra": deg, "dec": deg, "radius": deg, "limit": int?, "table": str?}
    ADQL        : {"adql": "SELECT ... FROM gaiadr3.gaia_source ..."}

The actual astroquery call lives in `_run_adql`, the single network seam. Tests
monkeypatch it with a fixture Table, so nothing here touches the network under test.
"""

from __future__ import annotations

from typing import Any

from astropy.table import Table

from .base import ensure_standard_columns

# Gaia DR3 source table and the columns we pull for a cone search.
DEFAULT_TABLE = "gaiadr3.gaia_source"
_CONE_COLUMNS = "source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax"


class GaiaSource:
    """Adapter over the ESA Gaia TAP service."""

    name = "gaia"

    def query(self, params: dict[str, Any]) -> Table:
        adql = params.get("adql") or self._cone_adql(params)
        table = self._run_adql(adql)
        return self._normalize(table)

    # -- query building ----------------------------------------------------
    def _cone_adql(self, params: dict[str, Any]) -> str:
        try:
            ra = float(params["ra"])
            dec = float(params["dec"])
            radius = float(params["radius"])
        except KeyError as e:
            raise ValueError(
                "cone search needs ra, dec, radius (deg) — or pass adql= instead"
            ) from e
        table = params.get("table", DEFAULT_TABLE)
        limit = int(params.get("limit", 2000))
        return (
            f"SELECT TOP {limit} {_CONE_COLUMNS} FROM {table} "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius}))"
        )

    # -- network seam (monkeypatched in tests) -----------------------------
    def _run_adql(self, adql: str) -> Table:  # pragma: no cover - network
        from astroquery.gaia import Gaia

        job = Gaia.launch_job_async(adql)
        return job.get_results()

    # -- normalization -----------------------------------------------------
    def _normalize(self, table: Table) -> Table:
        """Map Gaia columns onto the house standard set.

        Gaia already names its columns source_id/ra/dec, so normalization is mostly a
        validation pass — but do it explicitly so the seam holds if a query aliases
        columns differently.
        """
        out = table.copy()
        # Gaia's SOURCE_ID may come back upper-cased from some TAP endpoints.
        for cand in ("source_id", "SOURCE_ID"):
            if cand in out.colnames and cand != "source_id":
                out.rename_column(cand, "source_id")
        return ensure_standard_columns(out, require=True)
