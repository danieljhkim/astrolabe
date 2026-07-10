"""JPL Horizons adapter (M5, stretch) — via astroquery.jplhorizons.

Ephemerides are time-series, not a static catalog, so this adapter declares
`kind = "ephemeris"` and its datasets land under `data/processed/ephemeris/`
(SPEC §7, resolved). It returns the astropy Table Horizons gives, normalized only
to carry ra/dec + a source_id = target id.

    {"target": "499", "location": "500@399", "epochs": {...} | [jd, ...]}
"""

from __future__ import annotations

from typing import Any

from astropy.table import Table

from .base import ensure_standard_columns


class HorizonsSource:
    """Adapter over JPL Horizons ephemeris service."""

    name = "horizons"
    kind = "ephemeris"

    def query(self, params: dict[str, Any]) -> Table:
        table = self._run(params)
        return self._normalize(table, target=str(params.get("target", "")))

    # -- network seam (monkeypatched in tests) -----------------------------
    def _run(self, params: dict[str, Any]) -> Table:  # pragma: no cover - network
        from astroquery.jplhorizons import Horizons

        try:
            target = params["target"]
        except KeyError as e:
            raise ValueError("horizons query needs a 'target' id") from e
        obj = Horizons(
            id=target,
            location=params.get("location", "500@399"),
            epochs=params.get("epochs"),
        )
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
