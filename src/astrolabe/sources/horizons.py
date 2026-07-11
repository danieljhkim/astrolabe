"""JPL Horizons adapter (M5, stretch) — via astroquery.jplhorizons.

Ephemerides are time-series, not a static catalog, so this adapter declares
`kind = "ephemeris"` and its datasets land under `data/processed/ephemeris/`
(SPEC §7, resolved). Two query shapes:

Observer tables (default) — sky positions as seen from a location; the Table is
normalized to carry ra/dec + a source_id = target id:

    {"target": "499", "location": "500@399", "epochs": {...} | [jd, ...]}

State vectors (`"type": "vectors"`) — cartesian positions/velocities for
dynamics work (e.g. n-body baseline comparisons, ORB-10076). Defaults to a
heliocentric ICRF-aligned frame: origin `500@10` (Sun body center), refplane
"earth" (Earth mean equator of J2000). Columns are normalized to explicit-unit
house names; there is no ra/dec on this branch:

    {"target": "1", "type": "vectors",
     "epochs": {"start": "2016-01-01", "stop": "2026-01-01", "step": "10d"}}

    epoch_jd_tdb                        : Julian Date, TDB
    x_au, y_au, z_au                    : position, AU
    vx_au_d, vy_au_d, vz_au_d           : velocity, AU/day
"""

from __future__ import annotations

from typing import Any

from astropy.table import Table

from .base import ensure_standard_columns

# astroquery vectors-table column -> house name (explicit units).
_VECTOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("datetime_jd", "epoch_jd_tdb"),
    ("x", "x_au"),
    ("y", "y_au"),
    ("z", "z_au"),
    ("vx", "vx_au_d"),
    ("vy", "vy_au_d"),
    ("vz", "vz_au_d"),
)


class HorizonsSource:
    """Adapter over JPL Horizons ephemeris service."""

    name = "horizons"
    kind = "ephemeris"

    def query(self, params: dict[str, Any]) -> Table:
        table = self._run(params)
        target = str(params.get("target", ""))
        if params.get("type") == "vectors":
            return self._normalize_vectors(table, target=target)
        return self._normalize(table, target=target)

    # -- network seam (monkeypatched in tests) -----------------------------
    def _run(self, params: dict[str, Any]) -> Table:  # pragma: no cover - network
        from astroquery.jplhorizons import Horizons

        try:
            target = params["target"]
        except KeyError as e:
            raise ValueError("horizons query needs a 'target' id") from e
        vectors = params.get("type") == "vectors"
        obj = Horizons(
            id=target,
            location=params.get("location", "500@10" if vectors else "500@399"),
            epochs=params.get("epochs"),
        )
        if vectors:
            return obj.vectors(refplane=params.get("refplane", "earth"))
        return obj.ephemerides()

    # -- normalization -----------------------------------------------------
    def _normalize(self, table: Table, *, target: str) -> Table:
        out = table.copy()
        # Horizons uses RA/DEC (deg); expose lower-case house names.
        for src, dst in (("RA", "ra"), ("DEC", "dec")):
            if src in out.colnames and dst not in out.colnames:
                out.rename_column(src, dst)
        if "source_id" not in out.colnames:
            out["source_id"] = [target] * len(out)
        return ensure_standard_columns(out, require=True)

    def _normalize_vectors(self, table: Table, *, target: str) -> Table:
        out = table.copy()
        for src, dst in _VECTOR_COLUMNS:
            if src in out.colnames and dst not in out.colnames:
                out.rename_column(src, dst)
        missing = [dst for _, dst in _VECTOR_COLUMNS if dst not in out.colnames]
        if missing:
            raise ValueError(
                f"horizons vectors table is missing {missing}; has {list(out.colnames)}"
            )
        if "source_id" not in out.colnames:
            out["source_id"] = [target] * len(out)
        # No ra/dec on the vectors branch — cartesian state only.
        return out
