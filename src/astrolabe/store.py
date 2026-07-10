"""Catalog store: astropy Table <-> Parquet, with a DuckDB query layer.

Layout under `data/` (gitignored, SPEC §3):

    data/raw/<source>/<name>.parquet         # as fetched, pre-normalization
    data/processed/<kind>/<name>.parquet     # normalized dataset
    data/processed/<kind>/<name>.json        # sidecar metadata
    data/scratch/                            # notebook scratch space — not part of the
                                             # catalog, never queried, freely deletable

A dataset's directory says what kind of thing it is, its name says what it covers,
and its sidecar says exactly how to rebuild it. The kinds and their naming grammar:

    catalog/    sky-survey snapshots      <field>_<source>    (pleiades_gaia)
    ephemeris/  time-series               <target>_<span>     (mars_2026)
    derived/    products of analysis/     <field>_<recipe>    (field_eq180_xmatch)

For fetched kinds, `source` in the sidecar is the adapter slug. Derived datasets
instead use `source` for the recipe (e.g. "analysis.crossmatch") and record their
parent datasets in `lineage`: a list of `{"dataset": name, "fetched_at": ...}`
entries (`dataset` may be null, with a `note`, when an input was never persisted).

Dataset versioning (SPEC §7, resolved): overwrite-by-name. Every sidecar carries the
exact query, so a dataset is reproducible from the public source. The rule that makes
overwrite safe: a name must be deterministic w.r.t. its query — same query = same
name (a refresh), changed query = new name. Time-dependent identity belongs in the
name (`mars_2026`), never only in `fetched_at`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
from astropy.table import Table

DEFAULT_DATA_DIR = Path("data")

# Dataset kinds — the first level under data/processed/.
KINDS: tuple[str, ...] = ("catalog", "ephemeris", "derived")


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown dataset kind {kind!r}; known: {', '.join(KINDS)}")
    return kind


@dataclass(frozen=True)
class DatasetMeta:
    """Self-describing metadata for one stored dataset (the sidecar JSON)."""

    name: str
    kind: str
    source: str
    query: dict[str, Any]
    fetched_at: str  # ISO-8601 UTC
    n_rows: int
    columns: list[str]
    lineage: list[dict[str, Any]] | None = None  # derived only: parent datasets

    @classmethod
    def now(
        cls,
        *,
        name: str,
        kind: str,
        source: str,
        query: dict[str, Any],
        table: Table,
        lineage: list[dict[str, Any]] | None = None,
    ) -> DatasetMeta:
        return cls(
            name=name,
            kind=kind,
            source=source,
            query=query,
            fetched_at=datetime.now(UTC).isoformat(),
            n_rows=len(table),
            columns=list(table.colnames),
            lineage=lineage,
        )


class Store:
    """Filesystem catalog rooted at a data directory.

    Round-trip contract (M1): `write` a Table, `read` it back structurally intact;
    `query` runs SQL over the processed Parquet via DuckDB.
    """

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.scratch_dir = self.data_dir / "scratch"  # outside the catalog

    # -- paths -------------------------------------------------------------
    def _raw_path(self, source: str, name: str) -> Path:
        return self.raw_dir / source / f"{name}.parquet"

    def _processed_path(self, kind: str, name: str) -> Path:
        return self.processed_dir / kind / f"{name}.parquet"

    def _meta_path(self, kind: str, name: str) -> Path:
        return self.processed_dir / kind / f"{name}.json"

    def _resolve_kind(self, name: str, kind: str | None) -> str:
        """The kind a dataset lives under; searches all kinds when not given."""
        if kind is not None:
            return _check_kind(kind)
        hits = [k for k in KINDS if self._processed_path(k, name).exists()]
        if not hits:
            raise FileNotFoundError(
                f"no processed dataset {name!r} under {self.processed_dir}"
            )
        if len(hits) > 1:
            raise ValueError(
                f"dataset {name!r} exists under multiple kinds {hits}; pass kind="
            )
        return hits[0]

    # -- write -------------------------------------------------------------
    def write(
        self,
        table: Table,
        *,
        name: str,
        source: str,
        kind: str = "catalog",
        query: dict[str, Any] | None = None,
        raw: Table | None = None,
        lineage: list[dict[str, Any]] | None = None,
    ) -> DatasetMeta:
        """Persist a normalized Table as a processed dataset + sidecar metadata.

        `kind` places the dataset in the layout above (adapters carry the right one
        as `Source.kind`). Derived datasets should pass `lineage` naming their parent
        datasets. If `raw` is given it is stored under `data/raw/<source>/` as
        fetched; the normalized `table` is the queryable one.
        """
        _check_kind(kind)
        _write_parquet(table, self._processed_path(kind, name))

        if raw is not None:
            _write_parquet(raw, self._raw_path(source, name))

        meta = DatasetMeta.now(
            name=name, kind=kind, source=source, query=query or {}, table=table,
            lineage=lineage,
        )
        self._meta_path(kind, name).write_text(json.dumps(asdict(meta), indent=2))
        return meta

    # -- read --------------------------------------------------------------
    def read(self, name: str, kind: str | None = None) -> Table:
        """Read a processed dataset back as an astropy Table.

        `kind` is only needed when the same name exists under more than one kind.
        """
        kind = self._resolve_kind(name, kind)
        path = self._processed_path(kind, name)
        if not path.exists():
            raise FileNotFoundError(f"no processed dataset {name!r} at {path}")
        return Table.read(path)

    def read_meta(self, name: str, kind: str | None = None) -> DatasetMeta:
        kind = self._resolve_kind(name, kind)
        meta_path = self._meta_path(kind, name)
        if not meta_path.exists():
            raise FileNotFoundError(f"no metadata for dataset {name!r} at {meta_path}")
        return DatasetMeta(**json.loads(meta_path.read_text()))

    def datasets(self, kind: str | None = None) -> list[tuple[str, str]]:
        """(kind, name) pairs of all processed datasets, sorted."""
        kinds = (_check_kind(kind),) if kind is not None else KINDS
        found: list[tuple[str, str]] = []
        for k in kinds:
            kind_dir = self.processed_dir / k
            if kind_dir.exists():
                found.extend((k, p.stem) for p in kind_dir.glob("*.parquet"))
        return sorted(found)

    def list_datasets(self, kind: str | None = None) -> list[str]:
        """Names of all processed datasets, sorted."""
        return sorted(name for _, name in self.datasets(kind))

    # -- query -------------------------------------------------------------
    def query(self, sql: str) -> Table:
        """Run SQL over the processed catalog and return a Table.

        Each dataset is registered as a DuckDB view under a schema named after its
        kind — `SELECT ... FROM catalog.pleiades_gaia`, `FROM derived.<name>`. A
        name unique across kinds also gets an unqualified alias, so plain
        `FROM <dataset>` keeps working. `read_parquet('...')` works for ad-hoc paths.
        """
        pairs = self.datasets()
        name_counts = Counter(name for _, name in pairs)
        con = duckdb.connect(":memory:")
        try:
            for kind, name in pairs:
                path = self._processed_path(kind, name).resolve().as_posix()
                # CREATE VIEW can't take a prepared parameter, so inline the path
                # with single-quote escaping. Dataset names are filesystem stems.
                safe_path = path.replace("'", "''")
                con.execute(f'CREATE SCHEMA IF NOT EXISTS "{kind}"')
                con.execute(
                    f'CREATE VIEW "{kind}"."{name}" AS SELECT * FROM '
                    f"read_parquet('{safe_path}')"
                )
                if name_counts[name] == 1:
                    con.execute(
                        f'CREATE VIEW "{name}" AS SELECT * FROM '
                        f"read_parquet('{safe_path}')"
                    )
            arrow_tbl = con.execute(sql).to_arrow_table()
        finally:
            con.close()
        return Table.from_pandas(arrow_tbl.to_pandas())


def _write_parquet(table: Table, path: Path) -> None:
    """Serialize an astropy Table to Parquet via Arrow.

    astropy's native Parquet writer round-trips units in Parquet metadata; we use it
    directly so `Table.read` restores column dtypes and structure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="parquet", overwrite=True)


def read_parquet(path: Path | str) -> Table:
    """Read any Parquet file as an astropy Table (convenience for fixtures/tools)."""
    return Table.read(Path(path))


def parquet_schema(path: Path | str) -> list[str]:
    """Column names in a Parquet file without loading it fully."""
    return list(pq.read_schema(Path(path)).names)
