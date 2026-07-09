"""Catalog store: astropy Table <-> Parquet, with a DuckDB query layer.

Layout under `data/` (gitignored, SPEC §3):

    data/raw/<source>/<dataset>.parquet        # as fetched
    data/processed/<dataset>.parquet           # normalized
    data/processed/<dataset>.json              # sidecar metadata

A "dataset" is one fetch result. Metadata (source, query, fetched_at, n_rows) rides
in a sidecar JSON next to the Parquet so the catalog is self-describing without a DB.

Dataset versioning (SPEC §7): the default is overwrite-by-name — write the same
dataset name to replace it. `fetched_at` is recorded in the sidecar so append-style
partitioning can be layered on later without changing the read path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
from astropy.table import Table

DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class DatasetMeta:
    """Self-describing metadata for one stored dataset (the sidecar JSON)."""

    name: str
    source: str
    query: dict[str, Any]
    fetched_at: str  # ISO-8601 UTC
    n_rows: int
    columns: list[str]

    @classmethod
    def now(
        cls,
        *,
        name: str,
        source: str,
        query: dict[str, Any],
        table: Table,
    ) -> DatasetMeta:
        return cls(
            name=name,
            source=source,
            query=query,
            fetched_at=datetime.now(UTC).isoformat(),
            n_rows=len(table),
            columns=list(table.colnames),
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

    # -- paths -------------------------------------------------------------
    def _raw_path(self, source: str, name: str) -> Path:
        return self.raw_dir / source / f"{name}.parquet"

    def _processed_path(self, name: str) -> Path:
        return self.processed_dir / f"{name}.parquet"

    def _meta_path(self, name: str) -> Path:
        return self.processed_dir / f"{name}.json"

    # -- write -------------------------------------------------------------
    def write(
        self,
        table: Table,
        *,
        name: str,
        source: str,
        query: dict[str, Any] | None = None,
        raw: Table | None = None,
    ) -> DatasetMeta:
        """Persist a normalized Table as a processed dataset + sidecar metadata.

        If `raw` is given it is stored under `data/raw/<source>/` as fetched; the
        normalized `table` is the one that becomes queryable under `data/processed/`.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet(table, self._processed_path(name))

        if raw is not None:
            self._raw_path(source, name).parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(raw, self._raw_path(source, name))

        meta = DatasetMeta.now(name=name, source=source, query=query or {}, table=table)
        self._meta_path(name).write_text(json.dumps(asdict(meta), indent=2))
        return meta

    # -- read --------------------------------------------------------------
    def read(self, name: str) -> Table:
        """Read a processed dataset back as an astropy Table."""
        path = self._processed_path(name)
        if not path.exists():
            raise FileNotFoundError(f"no processed dataset {name!r} at {path}")
        return Table.read(path)

    def read_meta(self, name: str) -> DatasetMeta:
        meta_path = self._meta_path(name)
        if not meta_path.exists():
            raise FileNotFoundError(f"no metadata for dataset {name!r} at {meta_path}")
        return DatasetMeta(**json.loads(meta_path.read_text()))

    def list_datasets(self) -> list[str]:
        """Names of all processed datasets, sorted."""
        if not self.processed_dir.exists():
            return []
        return sorted(p.stem for p in self.processed_dir.glob("*.parquet"))

    # -- query -------------------------------------------------------------
    def query(self, sql: str) -> Table:
        """Run SQL over the processed catalog and return a Table.

        Each processed dataset is registered as a DuckDB view named after it, so a
        query can `SELECT ... FROM <dataset>`. `read_parquet('...')` also works for
        ad-hoc paths.
        """
        con = duckdb.connect(":memory:")
        try:
            for ds in self.list_datasets():
                path = self._processed_path(ds).resolve().as_posix()
                # CREATE VIEW can't take a prepared parameter, so inline the path
                # with single-quote escaping. Dataset names are filesystem stems.
                safe_path = path.replace("'", "''")
                con.execute(
                    f'CREATE VIEW "{ds}" AS SELECT * FROM '
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
