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

Mutable dataset versioning (SPEC §7): overwrite-by-name. A name must be deterministic
w.r.t. its query — same query = same name (a refresh), changed query = new name.
Time-dependent identity belongs in the name (`mars_2026`), never only in `fetched_at`.
Reproducible scientific consumption first calls ``Store.snapshot``; content-addressed
Parquet + exact sidecar bytes survive later alias overwrites and carry exact parent pins.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow.parquet as pq
from astropy.table import Table

if TYPE_CHECKING:
    from .provenance import Snapshot

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
    # column name -> meaning + units, for datasets consumed outside this repo
    # (exchange deliveries à la ORB-10075/ORB-10168)
    semantics: dict[str, str] | None = None
    # New writes bind the sidecar to the exact Parquet generation. Older sidecars
    # omit these fields and remain readable; snapshot capture handles that legacy
    # case conservatively by copying and hashing both files under a shared lock.
    parquet_sha256: str | None = None
    schema: list[dict[str, Any]] | None = None
    units: dict[str, str] | None = None

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
        semantics: dict[str, str] | None = None,
        parquet_sha256: str | None = None,
        schema: list[dict[str, Any]] | None = None,
        units: dict[str, str] | None = None,
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
            semantics=semantics,
            parquet_sha256=parquet_sha256,
            schema=schema,
            units=units,
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
        self.snapshot_dir = self.data_dir / "snapshots" / "v1" / "sha256"

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
            raise FileNotFoundError(f"no processed dataset {name!r} under {self.processed_dir}")
        if len(hits) > 1:
            raise ValueError(f"dataset {name!r} exists under multiple kinds {hits}; pass kind=")
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
        semantics: dict[str, str] | None = None,
    ) -> DatasetMeta:
        """Persist a normalized Table as a processed dataset + sidecar metadata.

        `kind` places the dataset in the layout above (adapters carry the right one
        as `Source.kind`). Derived datasets should pass `lineage` naming their parent
        datasets. If `raw` is given it is stored under `data/raw/<source>/` as
        fetched; the normalized `table` is the queryable one. `semantics` (column
        name -> meaning + units) makes the sidecar self-describing for datasets
        consumed outside this repo.
        """
        _check_kind(kind)
        parquet_path = self._processed_path(kind, name)
        meta_path = self._meta_path(kind, name)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        staged_parquet = _temporary_path(parquet_path)
        staged_meta = _temporary_path(meta_path)
        staged_raw: Path | None = None
        raw_path = self._raw_path(source, name)
        try:
            _write_parquet(table, staged_parquet)
            parquet_digest = _file_digest(staged_parquet)
            meta = DatasetMeta.now(
                name=name,
                kind=kind,
                source=source,
                query=query or {},
                table=table,
                lineage=lineage,
                semantics=semantics,
                parquet_sha256=parquet_digest,
                schema=_table_schema(table),
                units=_table_units(table),
            )
            staged_meta.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
            if raw is not None:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                staged_raw = _temporary_path(raw_path)
                _write_parquet(raw, staged_raw)

            with self._catalog_lock(exclusive=True):
                _replace_pair(
                    staged_parquet,
                    parquet_path,
                    staged_meta,
                    meta_path,
                )
                if staged_raw is not None:
                    os.replace(staged_raw, raw_path)
            return meta
        finally:
            for path in (staged_parquet, staged_meta, staged_raw):
                if path is not None:
                    path.unlink(missing_ok=True)

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
                    f'CREATE VIEW "{kind}"."{name}" AS SELECT * FROM read_parquet(\'{safe_path}\')'
                )
                if name_counts[name] == 1:
                    con.execute(
                        f"CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet('{safe_path}')"
                    )
            arrow_tbl = con.execute(sql).to_arrow_table()
        finally:
            con.close()
        return Table.from_pandas(arrow_tbl.to_pandas())

    # -- immutable scientific snapshots ----------------------------------
    def snapshot(
        self,
        name: str,
        kind: str | None = None,
        *,
        parents: list[Path | str] | None = None,
        allow_unresolved: bool = False,
    ) -> Snapshot:
        """Retain an immutable, content-addressed dataset snapshot.

        ``parents`` are orbit-research v1 artifact record paths, not mutable
        dataset names. Derived lineage without an exact supplied parent is rejected
        unless ``allow_unresolved`` is explicitly selected for historical migration.
        """
        from .provenance import capture_snapshot

        return capture_snapshot(
            self,
            name,
            kind,
            parents=parents or [],
            allow_unresolved=allow_unresolved,
        )

    def read_snapshot(self, digest: str) -> Table:
        """Verify and read an immutable snapshot by its ``sha256:...`` pin."""
        from .provenance import read_snapshot

        return read_snapshot(self, digest)

    def restore_snapshot(
        self,
        digest: str,
        *,
        overwrite: bool = False,
    ) -> DatasetMeta:
        """Restore a retained generation to its mutable kind/name alias."""
        from .provenance import restore_snapshot

        return restore_snapshot(self, digest, overwrite=overwrite)

    @contextmanager
    def _catalog_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Coordinate writers and snapshot readers without changing data layout."""
        import fcntl

        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.data_dir / ".astrolabe-catalog.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _write_parquet(table: Table, path: Path) -> None:
    """Serialize an astropy Table to Parquet via Arrow.

    astropy's native Parquet writer round-trips units in Parquet metadata; we use it
    directly so `Table.read` restores column dtypes and structure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="parquet", overwrite=True)


def _temporary_path(target: Path) -> Path:
    """Reserve a same-directory temporary name for an atomic rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    path = Path(raw_path)
    path.unlink()
    return path


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _table_schema(table: Table) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "dtype": str(table[name].dtype),
            "shape": list(table[name].shape[1:]),
        }
        for name in table.colnames
    ]


def _table_units(table: Table) -> dict[str, str]:
    return {
        name: str(table[name].unit)
        for name in table.colnames
        if getattr(table[name], "unit", None) is not None
    }


def _replace_pair(
    staged_data: Path,
    data_path: Path,
    staged_meta: Path,
    meta_path: Path,
) -> None:
    """Publish a dataset pair and restore the old generation on partial failure."""
    backups: dict[Path, Path] = {}
    for current in (data_path, meta_path):
        if current.exists():
            backup = _temporary_path(current)
            shutil.copyfile(current, backup)
            backups[current] = backup
    replaced: list[Path] = []
    try:
        os.replace(staged_data, data_path)
        replaced.append(data_path)
        os.replace(staged_meta, meta_path)
        replaced.append(meta_path)
    except BaseException:
        for current in reversed(replaced):
            backup = backups.pop(current, None)
            if backup is None:
                current.unlink(missing_ok=True)
            else:
                os.replace(backup, current)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def read_parquet(path: Path | str) -> Table:
    """Read any Parquet file as an astropy Table (convenience for fixtures/tools)."""
    return Table.read(Path(path))


def parquet_schema(path: Path | str) -> list[str]:
    """Column names in a Parquet file without loading it fully."""
    return list(pq.read_schema(Path(path)).names)
