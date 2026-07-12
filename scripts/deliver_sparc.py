"""Deliver the SPARC datasets for the fixed-length-scale universality test (ORB-10168).

Fetches the two published SPARC tables (network!), writes the full catalog, then
builds the quality-flagged analysis subsample with the standard
radial-acceleration-relation arrays:

    catalog/sparc_galaxies     per-galaxy properties (175 late-type galaxies)
    catalog/sparc_rotcurves    per-radius mass-model rotation curves
    derived/sparc_rar          Q <= 2, incl >= 30 deg subsample with g_obs/g_bar
                               (dwarfs through giant spirals — SPARC spans them)

Each sidecar carries `semantics` (column -> meaning + units) so the ws_orrery
fit apparatus can import the delivery unchanged; the exchange contract is
docs/sparc-interface.md.

Usage: uv run python scripts/deliver_sparc.py [--data-dir data]
"""

from __future__ import annotations

import argparse

from astrolabe.analysis.rar import COLUMN_SEMANTICS as RAR_SEMANTICS
from astrolabe.analysis.rar import radial_acceleration_relation
from astrolabe.sources import get_source
from astrolabe.sources.sparc import COLUMN_SEMANTICS
from astrolabe.store import Store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    sparc = get_source("sparc")
    store = Store(args.data_dir)

    galaxies = sparc.query({"table": "galaxies"})
    meta = store.write(
        galaxies,
        name="sparc_galaxies",
        source=sparc.name,
        kind=sparc.kind,
        query={"table": "galaxies"},
        semantics=COLUMN_SEMANTICS["galaxies"],
    )
    print(f"wrote {meta.kind}/{meta.name}: {meta.n_rows} galaxies")

    rotcurves = sparc.query({"table": "rotation_curves"})
    meta = store.write(
        rotcurves,
        name="sparc_rotcurves",
        source=sparc.name,
        kind=sparc.kind,
        query={"table": "rotation_curves"},
        semantics=COLUMN_SEMANTICS["rotation_curves"],
    )
    print(f"wrote {meta.kind}/{meta.name}: {meta.n_rows} rotation-curve points")

    rar = radial_acceleration_relation(rotcurves, galaxies)
    galaxies_meta = store.read_meta("sparc_galaxies")
    rotcurves_meta = store.read_meta("sparc_rotcurves")
    meta = store.write(
        rar,
        name="sparc_rar",
        source="analysis.radial_acceleration_relation",
        kind="derived",
        query=dict(rar.meta["rar"]),
        lineage=[
            {"dataset": "sparc_rotcurves", "fetched_at": rotcurves_meta.fetched_at},
            {"dataset": "sparc_galaxies", "fetched_at": galaxies_meta.fetched_at},
        ],
        semantics={**COLUMN_SEMANTICS["rotation_curves"], **RAR_SEMANTICS},
    )
    n_gal = len(set(rar["source_id"]))
    print(f"wrote {meta.kind}/{meta.name}: {meta.n_rows} points, {n_gal} galaxies "
          f"(Q <= 2, incl >= 30 deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
