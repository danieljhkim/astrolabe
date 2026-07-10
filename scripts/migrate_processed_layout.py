"""One-shot migration: flat data/processed/ -> kind-partitioned layout (2026-07).

Moves each top-level <name>.parquet + <name>.json pair into
data/processed/<kind>/, adds `kind` (and `lineage` for known derived datasets)
to the sidecar, and applies the naming grammar to datasets that predate it
(see store.py's module docstring for the layout and grammar).

Idempotent: files already under a kind directory are untouched; a re-run on a
migrated catalog is a no-op. Run from the repo root:

    uv run python scripts/migrate_processed_layout.py [data-dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Datasets known at migration time (2026-07) that need more than kind inference.
RENAMES = {
    "mars_ephem_2026": "mars_2026",  # ephemeris grammar: <target>_<span>
}
DERIVED_FIXUPS: dict[str, dict] = {
    "field_eq180_xmatch": {
        # was source="gaia+sdss" — a fake provider slug on a derived product
        "source": "analysis.crossmatch",
        "lineage": [
            {"dataset": "field_eq180_sdss"},
            {"dataset": None, "source": "gaia",
             "note": "gaia input was not persisted as a dataset"},
        ],
    },
}


def infer_kind(name: str, source: str) -> str:
    if name in DERIVED_FIXUPS or "+" in source:
        return "derived"
    if source == "horizons":
        return "ephemeris"
    return "catalog"


def migrate(processed: Path) -> int:
    moved = 0
    for parquet in sorted(processed.glob("*.parquet")):  # top level only = legacy
        old_name = parquet.stem
        sidecar = processed / f"{old_name}.json"
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {"name": old_name}

        kind = infer_kind(old_name, meta.get("source", ""))
        new_name = RENAMES.get(old_name, old_name)
        meta["name"] = new_name
        meta["kind"] = kind
        meta.setdefault("lineage", None)
        meta.update(DERIVED_FIXUPS.get(old_name, {}))

        # Fill lineage fetched_at from the parent's sidecar when resolvable.
        for entry in meta["lineage"] or []:
            parent = entry.get("dataset")
            if parent and "fetched_at" not in entry:
                for candidate in (processed / f"{parent}.json",
                                  *processed.glob(f"*/{parent}.json")):
                    if candidate.exists():
                        entry["fetched_at"] = json.loads(
                            candidate.read_text())["fetched_at"]
                        break

        kind_dir = processed / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        parquet.rename(kind_dir / f"{new_name}.parquet")
        (kind_dir / f"{new_name}.json").write_text(json.dumps(meta, indent=2))
        if sidecar.exists():
            sidecar.unlink()
        print(f"{old_name} -> {kind}/{new_name}")
        moved += 1
    return moved


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    processed = data_dir / "processed"
    if not processed.exists():
        print(f"nothing to migrate: {processed} does not exist")
        raise SystemExit(0)
    n = migrate(processed)
    print(f"migrated {n} dataset(s)" if n else "nothing to migrate (already kind-partitioned)")
