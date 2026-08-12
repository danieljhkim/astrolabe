"""Wide-binary local-volume star sample via Gaia DR3 TAP/ADQL.

This adapter fetches a *star* catalog suitable for El-Badry-style pairing in
`astrolabe.analysis.wide_binaries` — it does not return pairs itself. Pairing,
chance-alignment control, and ṽ(g_N) live in analysis/ (pure functions).

Query modes:

    default / local volume :
        {"mode": "local_volume", "limit": int?, "parallax_min_mas": float?,
         "ruwe_max": float?, "parallax_over_error_min": float?,
         "g_mag_max": float?}
    ADQL passthrough :
        {"adql": "SELECT ... FROM gaiadr3.gaia_source ..."}

The network call is isolated in `_run_adql` so tests monkeypatch a fixture.
"""

from __future__ import annotations

from typing import Any

from astropy.table import Table

from .base import ensure_standard_columns

DEFAULT_TABLE = "gaiadr3.gaia_source"

# Columns required by analysis.wide_binaries.select_wide_pairs.
_COLUMNS = (
    "source_id, ra, dec, parallax, parallax_error, "
    "pmra, pmra_error, pmdec, pmdec_error, "
    "phot_g_mean_mag, bp_rp, ruwe, "
    "radial_velocity, radial_velocity_error"
)

# Canonical ADQL template for the ORB-10753 local-volume sample. Exact string
# with resolved parameters rides in the store sidecar `query`.
#
# All-sky random TOP-N is too sparse for pair finding (nn ≫ θ_max(s_max)).
# Default is a large ICRS cone so neighbours actually sit within the projected
# separation window; footprint (ra0, dec0, radius_deg) is part of the query
# identity. Pass adql= for a custom footprint / full-sky chunked campaign.
LOCAL_VOLUME_ADQL_TEMPLATE = """SELECT {top_clause}
{columns}
FROM {table}
WHERE parallax > {parallax_min_mas}
  AND parallax_over_error > {parallax_over_error_min}
  AND ruwe < {ruwe_max}
  AND phot_g_mean_mag < {g_mag_max}
  AND bp_rp IS NOT NULL
  AND pmra IS NOT NULL AND pmdec IS NOT NULL
  AND 1=CONTAINS(POINT('ICRS', ra, dec),
                 CIRCLE('ICRS', {ra0}, {dec0}, {radius_deg}))
"""


class WideBinarySource:
    """Gaia DR3 local-volume star sample for wide-binary pairing."""

    name = "wide_binaries"
    kind = "catalog"

    def query(self, params: dict[str, Any]) -> Table:
        adql = params.get("adql") or self._local_volume_adql(params)
        table = self._run_adql(adql)
        out = self._normalize(table)
        out.meta["wide_binaries_query"] = {
            "adql": adql,
            "params": {k: v for k, v in params.items() if k != "adql"},
            "doi_gaia_dr3": "10.5270/esa-1ugzkg7",
            "reference_selection": (
                "El-Badry, Rix & Heintz 2021, MNRAS 506, 2269 "
                "(style; cuts predeclared in ORB-10753 plan)"
            ),
        }
        return out

    def _local_volume_adql(self, params: dict[str, Any]) -> str:
        # limit=0 or None → no TOP (return the full cone).
        limit = params.get("limit", 0)
        top_clause = f"TOP {int(limit)}" if limit not in (None, 0, "0") else ""
        parallax_min_mas = float(params.get("parallax_min_mas", 5.0))
        ruwe_max = float(params.get("ruwe_max", 1.4))
        poe_min = float(params.get("parallax_over_error_min", 10.0))
        g_mag_max = float(params.get("g_mag_max", 18.0))
        # Default footprint: mid-latitude northern field, 30 deg radius — dense
        # enough for s ≲ 50 kau pairs inside d < 200 pc.
        ra0 = float(params.get("ra0", 180.0))
        dec0 = float(params.get("dec0", 40.0))
        radius_deg = float(params.get("radius_deg", 30.0))
        table = params.get("table", DEFAULT_TABLE)
        return LOCAL_VOLUME_ADQL_TEMPLATE.format(
            top_clause=top_clause,
            columns=_COLUMNS,
            table=table,
            parallax_min_mas=parallax_min_mas,
            parallax_over_error_min=poe_min,
            ruwe_max=ruwe_max,
            g_mag_max=g_mag_max,
            ra0=ra0,
            dec0=dec0,
            radius_deg=radius_deg,
        )

    def _run_adql(self, adql: str) -> Table:  # pragma: no cover - network
        from astroquery.gaia import Gaia

        job = Gaia.launch_job_async(adql)
        return job.get_results()

    def _normalize(self, table: Table) -> Table:
        out = table.copy()
        # TAP may uppercase column names.
        rename = {}
        lower_map = {c.lower(): c for c in out.colnames}
        for want in (
            "source_id", "ra", "dec", "parallax", "parallax_error",
            "pmra", "pmra_error", "pmdec", "pmdec_error",
            "phot_g_mean_mag", "bp_rp", "ruwe",
            "radial_velocity", "radial_velocity_error",
        ):
            if want not in out.colnames and want in lower_map:
                rename[lower_map[want]] = want
        for old, new in rename.items():
            out.rename_column(old, new)
        if "SOURCE_ID" in out.colnames and "source_id" not in out.colnames:
            out.rename_column("SOURCE_ID", "source_id")
        return ensure_standard_columns(out, require=True)
