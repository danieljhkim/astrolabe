"""Immutable dataset snapshots using the orbit-research v1 artifact contract.

This module is the Astrolabe owner adapter.  Generic record identity, revision
hashing, references, manifests, and validation remain in ``orbit_research``.
Large Parquet bytes remain under the ignored data root; tracked research records
contain exact pins and legacy metadata, never the dataset itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from base64 import b64encode
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from astropy.table import Table
from orbit_research import make_record, validate
from orbit_research.contract import canonical, digest_bytes, reference, revision_digest

from .store import DatasetMeta, Store, _file_digest, _replace_pair, _temporary_path

FRAMEWORK_VERSION = "0.1.0"
FRAMEWORK_REVISION = "7b6c1b2380bc915d6ff7cca50f288ed716a99c74"
SNAPSHOT_MEDIA_TYPE = "application/vnd.apache.parquet"


class SnapshotError(ValueError):
    """A snapshot could not be proven complete and internally consistent."""


class SnapshotConflictError(SnapshotError):
    """A content-addressed destination already contains conflicting content."""


class SourceChangedError(SnapshotError):
    """Mutable source bytes changed during capture."""


@lru_cache(maxsize=1)
def _framework_check() -> None:
    try:
        dist = distribution("orbit-research")
    except PackageNotFoundError as exc:
        raise SnapshotError(
            "install the pinned orbit-research dependency with the research extra"
        ) from exc
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if not (
        dist.version == FRAMEWORK_VERSION
        and direct.get("vcs_info", {}).get("commit_id") == FRAMEWORK_REVISION
    ):
        raise SnapshotError(
            "orbit-research must be the exact revision pinned by the research extra"
        )


@dataclass(frozen=True)
class Snapshot:
    """One verified immutable dataset generation."""

    digest: str
    directory: Path
    record: dict[str, Any]
    created: bool

    @property
    def data_path(self) -> Path:
        return self.directory / "dataset.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def record_path(self) -> Path:
        return self.directory / "record.json"


def _strict_object(data: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise SnapshotError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{label} must contain a JSON object")
    return value


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _git_revision() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parquet_facts(path: Path) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    schema = pq.read_schema(path)
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    metadata = {
        key.decode("utf-8", errors="surrogateescape"): digest_bytes(value)
        for key, value in sorted((schema.metadata or {}).items())
    }
    return fields, metadata, pq.ParquetFile(path).metadata.num_rows


def _normal_digest(value: str) -> str:
    if value.startswith("sha256:"):
        value = value.removeprefix("sha256:")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise SnapshotError("snapshot digest must be sha256:<64 lowercase hex characters>")
    return f"sha256:{value}"


def _load_parent(path: Path | str) -> dict[str, Any]:
    record_path = Path(path)
    record = _strict_object(record_path.read_bytes(), label=str(record_path))
    errors = validate(record)
    if errors:
        raise SnapshotError(f"invalid parent record {record_path}: {'; '.join(errors)}")
    if record["kind"] != "artifact" or record["payload"]["role"] != "dataset":
        raise SnapshotError(f"parent {record_path} is not a dataset artifact")
    if record["payload"]["availability"] != "available":
        raise SnapshotError(f"parent {record_path} is not available")
    # A local snapshot record must still match the retained bytes beside it.
    if record_path.name == "record.json":
        _verify_snapshot_directory(record_path.parent, record)
    return record


def _lineage_resolution(
    lineage: Any,
    parents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if lineage is None:
        if parents:
            raise SnapshotError("parent pins supplied but sidecar records no lineage")
        return [], []
    if not isinstance(lineage, list):
        raise SnapshotError("sidecar lineage must be null or a list")

    by_name: dict[str, list[dict[str, Any]]] = {}
    for parent in parents:
        dataset = parent.get("legacy", {}).get("dataset", {})
        name = dataset.get("name") if isinstance(dataset, dict) else None
        if not isinstance(name, str):
            raise SnapshotError("parent record does not preserve legacy dataset name")
        by_name.setdefault(name, []).append(parent)

    resolutions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for index, entry in enumerate(lineage):
        if not isinstance(entry, dict):
            resolutions.append(
                {
                    "index": index,
                    "legacy": entry,
                    "status": "pending",
                    "reason": "Malformed legacy lineage entry retained verbatim.",
                }
            )
            continue
        name = entry.get("dataset")
        matches = by_name.get(name, []) if isinstance(name, str) else []
        if len(matches) == 1:
            parent = matches[0]
            pin = reference(parent)
            references.append(pin)
            used.add((parent["id"], parent["revision_id"]))
            resolutions.append(
                {
                    "index": index,
                    "legacy": entry,
                    "status": "current-pin",
                    "reference": pin,
                    "limitation": (
                        "The pin proves the explicitly supplied current parent; the "
                        "legacy name/timestamp alone does not prove historical consumption."
                    ),
                }
            )
        else:
            reason = (
                "Legacy parent was never persisted."
                if name is None
                else "No unique immutable parent pin was supplied for this legacy name."
            )
            resolutions.append(
                {"index": index, "legacy": entry, "status": "pending", "reason": reason}
            )
    unused = [
        parent["id"] for parent in parents if (parent["id"], parent["revision_id"]) not in used
    ]
    if unused:
        raise SnapshotError(f"parent pins do not match sidecar lineage: {unused}")
    return resolutions, references


def capture_snapshot(
    store: Store,
    name: str,
    kind: str | None = None,
    *,
    parents: Iterable[Path | str] = (),
    allow_unresolved: bool = False,
) -> Snapshot:
    """Copy a stable mutable pair into the content-addressed snapshot store."""
    _framework_check()
    kind = store._resolve_kind(name, kind)
    data_path = store._processed_path(kind, name)
    meta_path = store._meta_path(kind, name)
    parent_records = [_load_parent(path) for path in parents]
    staging_root = store.data_dir / "snapshots" / "v1" / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="capture-", dir=staging_root))
    staged_data = staging / "dataset.parquet"
    staged_meta = staging / "metadata.json"
    try:
        with store._catalog_lock(exclusive=False):
            if not data_path.is_file() or not meta_path.is_file():
                raise SnapshotError(f"dataset pair is incomplete for {kind}/{name}")
            meta_before = meta_path.read_bytes()
            shutil.copyfile(data_path, staged_data)
            shutil.copyfile(meta_path, staged_meta)
            data_digest = _file_digest(staged_data)
            # Full second reads catch uncoordinated external writers as well as the
            # Store lock coordinating all current Astrolabe writers.
            if meta_path.read_bytes() != meta_before or _file_digest(data_path) != data_digest:
                raise SourceChangedError(f"dataset changed during snapshot: {kind}/{name}")

        raw_meta = _strict_object(meta_before, label=str(meta_path))
        if raw_meta.get("name") != name or raw_meta.get("kind") != kind:
            raise SnapshotError("sidecar identity differs from its mutable kind/name path")
        expected_digest = raw_meta.get("parquet_sha256")
        if expected_digest is not None and expected_digest != data_digest:
            raise SourceChangedError("sidecar Parquet digest does not match dataset bytes")

        schema, parquet_metadata, n_rows = _parquet_facts(staged_data)
        columns = raw_meta.get("columns")
        if columns != [field["name"] for field in schema] or raw_meta.get("n_rows") != n_rows:
            raise SnapshotError("sidecar row/column metadata does not match Parquet schema")
        resolutions, references = _lineage_resolution(raw_meta.get("lineage"), parent_records)
        unresolved = [item for item in resolutions if item["status"] == "pending"]
        if unresolved and not allow_unresolved:
            raise SnapshotError(
                "lineage has unresolved parents; supply exact parent records or use "
                "allow_unresolved only for an explicitly limited historical capture"
            )

        evidence = {
            "format": "astrolabe-dataset-snapshot-v1",
            "dataset": {"kind": kind, "name": name},
            "parquet_sha256": data_digest,
            "sidecar_sha256": digest_bytes(meta_before),
            "schema": schema,
            "parquet_metadata": parquet_metadata,
            "units": raw_meta.get("units") or {},
            "parent_pins": references,
        }
        snapshot_digest = digest_bytes(canonical(evidence))
        hex_digest = snapshot_digest.removeprefix("sha256:")
        target = store.snapshot_dir / hex_digest
        relative_meta = (target / "metadata.json").relative_to(store.data_dir).as_posix()
        revision = _git_revision()
        missingness: list[str] = ["non-git-dataset-bytes"]
        if raw_meta.get("lineage") is None:
            missingness.append("lineage-not-recorded")
        if unresolved:
            missingness.append("historical-lineage-unresolved")
        provenance = {
            "repository": "astrolabe",
            "git_revision": revision,
            "blob_oid": None,
            "sha256": digest_bytes(meta_before),
            "path": relative_meta,
            "selector": "$",
            "historical": False,
            "working_tree": True,
        }
        record = make_record(
            "astrolabe",
            "artifact",
            f"dataset:{kind}:{name}",
            {
                "role": "dataset",
                "availability": "available",
                "snapshot_digest": snapshot_digest,
                "locator": (target / "dataset.parquet").relative_to(store.data_dir).as_posix(),
                "media_type": SNAPSHOT_MEDIA_TYPE,
            },
            provenance,
            legacy={
                "dataset": raw_meta,
                "snapshot": evidence,
                "lineage_resolution": resolutions,
            },
            activity="active",
            scope="unknown",
            limitations=[
                "Captured bytes prove current availability, not which bytes a "
                "historical run consumed.",
                "Non-Git dataset retention is owner-local; copy the complete "
                "snapshot directory when exporting bytes.",
            ],
            missingness=missingness,
        )
        record["aliases"] = [f"{kind}/{name}", name]
        record["references"] = references
        record["revision_id"] = revision_digest(record)
        errors = validate(record)
        if errors:
            raise SnapshotError("orbit-research rejected snapshot record: " + "; ".join(errors))
        record_bytes = _encoded(record)
        (staging / "record.json").write_bytes(record_bytes)

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging, target)
            created = True
        except OSError:
            if not target.is_dir():
                raise
            record = _verify_existing(target, staged_data, meta_before, record)
            created = False
        result = Snapshot(snapshot_digest, target, record, created)
        _verify_snapshot_directory(target, record)
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _verify_existing(
    target: Path,
    staged_data: Path,
    meta_bytes: bytes,
    proposed_record: dict[str, Any],
) -> dict[str, Any]:
    record_path = target / "record.json"
    if not record_path.is_file():
        raise SnapshotConflictError(
            f"snapshot destination conflicts with content-addressed pin {target.name}"
        )
    existing = _strict_object(record_path.read_bytes(), label=str(record_path))
    try:
        errors = validate(existing)
        if errors:
            raise SnapshotConflictError("; ".join(errors))
        _verify_snapshot_directory(target, existing)
    except (KeyError, TypeError, SnapshotError) as exc:
        raise SnapshotConflictError(
            f"snapshot destination conflicts with content-addressed pin {target.name}"
        ) from exc
    if (
        _file_digest(target / "dataset.parquet") != _file_digest(staged_data)
        or _file_digest(target / "metadata.json") != digest_bytes(meta_bytes)
        or existing["id"] != proposed_record["id"]
        or existing["revision_id"] != proposed_record["revision_id"]
    ):
        raise SnapshotConflictError(
            f"snapshot destination conflicts with content-addressed pin {target.name}"
        )
    return existing


def _verify_snapshot_directory(target: Path, record: dict[str, Any]) -> None:
    data_path = target / "dataset.parquet"
    meta_path = target / "metadata.json"
    record_path = target / "record.json"
    if not all(path.is_file() for path in (data_path, meta_path, record_path)):
        raise SnapshotConflictError(f"incomplete retained snapshot at {target}")
    on_disk_record = _strict_object(record_path.read_bytes(), label=str(record_path))
    if on_disk_record != record:
        raise SnapshotConflictError(f"snapshot record changed at {record_path}")
    evidence = record.get("legacy", {}).get("snapshot")
    if not isinstance(evidence, dict):
        raise SnapshotConflictError("snapshot record lacks owner evidence")
    if _file_digest(data_path) != evidence.get("parquet_sha256"):
        raise SnapshotConflictError("retained Parquet bytes do not match snapshot record")
    if _file_digest(meta_path) != evidence.get("sidecar_sha256"):
        raise SnapshotConflictError("retained sidecar bytes do not match snapshot record")
    if digest_bytes(canonical(evidence)) != record["payload"]["snapshot_digest"]:
        raise SnapshotConflictError("snapshot evidence no longer matches its digest")


def load_snapshot(store: Store, digest: str) -> Snapshot:
    digest = _normal_digest(digest)
    target = store.snapshot_dir / digest.removeprefix("sha256:")
    record_path = target / "record.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"no retained snapshot for {digest}")
    record = _strict_object(record_path.read_bytes(), label=str(record_path))
    errors = validate(record)
    if errors:
        raise SnapshotConflictError("invalid retained snapshot record: " + "; ".join(errors))
    if record["payload"].get("snapshot_digest") != digest:
        raise SnapshotConflictError("snapshot directory name differs from record digest")
    _verify_snapshot_directory(target, record)
    return Snapshot(digest, target, record, False)


def read_snapshot(store: Store, digest: str) -> Table:
    return Table.read(load_snapshot(store, digest).data_path)


def restore_snapshot(store: Store, digest: str, *, overwrite: bool = False) -> DatasetMeta:
    snapshot = load_snapshot(store, digest)
    raw_meta = _strict_object(snapshot.metadata_path.read_bytes(), label="snapshot metadata")
    name = raw_meta.get("name")
    kind = raw_meta.get("kind")
    if not isinstance(name, str) or not isinstance(kind, str):
        raise SnapshotConflictError("snapshot metadata lacks dataset identity")
    data_path = store._processed_path(kind, name)
    meta_path = store._meta_path(kind, name)
    if not overwrite and (data_path.exists() or meta_path.exists()):
        raise FileExistsError(f"mutable dataset {kind}/{name} already exists")
    staged_data = _temporary_path(data_path)
    staged_meta = _temporary_path(meta_path)
    try:
        shutil.copyfile(snapshot.data_path, staged_data)
        shutil.copyfile(snapshot.metadata_path, staged_meta)
        with store._catalog_lock(exclusive=True):
            _replace_pair(staged_data, data_path, staged_meta, meta_path)
    finally:
        staged_data.unlink(missing_ok=True)
        staged_meta.unlink(missing_ok=True)
    return DatasetMeta(**raw_meta)


def make_manifest(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    _framework_check()
    records = list(records)
    for record in records:
        if "historical-lineage-unresolved" in record.get("missingness", []):
            raise SnapshotError(
                f"{record['id']} has unresolved lineage and cannot be exported "
                "as a reproducible run input"
            )
    revisions = sorted(
        {
            (record["provenance"]["repository"], record["provenance"]["git_revision"])
            for record in records
        },
        key=lambda item: (item[0], item[1] or ""),
    )
    manifest = {
        "schema_version": 1,
        "kind": "manifest",
        "repositories": [
            {"id": repository, "git_revision": revision} for repository, revision in revisions
        ],
        "references": [reference(record) for record in records],
    }
    errors = validate(manifest)
    if errors:
        raise SnapshotError("orbit-research rejected manifest: " + "; ".join(errors))
    return manifest


def export_manifest(record_paths: Iterable[Path | str], output: Path | str) -> Path:
    records = [_load_parent(path) for path in record_paths]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_encoded(make_manifest(records)))
    return path


def trace_snapshot(store: Store, digest: str) -> dict[str, Any]:
    root = load_snapshot(store, digest)
    index: dict[tuple[str, str], Snapshot] = {}
    if store.snapshot_dir.exists():
        for record_path in sorted(store.snapshot_dir.glob("*/record.json")):
            record = _strict_object(record_path.read_bytes(), label=str(record_path))
            errors = validate(record)
            if errors:
                raise SnapshotConflictError(
                    f"invalid traced record {record_path}: {'; '.join(errors)}"
                )
            traced = Snapshot(
                record["payload"]["snapshot_digest"], record_path.parent, record, False
            )
            _verify_snapshot_directory(record_path.parent, record)
            index[(record["id"], record["revision_id"])] = traced

    seen: set[tuple[str, str]] = set()

    def visit(snapshot: Snapshot) -> dict[str, Any]:
        record = snapshot.record
        key = (record["id"], record["revision_id"])
        if key in seen:
            return {"id": key[0], "revision_id": key[1], "cycle": True}
        seen.add(key)
        children = []
        for pin in record["references"]:
            child = index.get((pin["id"], pin["revision_id"]))
            children.append(visit(child) if child else {**pin, "missing_local_snapshot": True})
        return {
            "id": record["id"],
            "revision_id": record["revision_id"],
            "snapshot_digest": snapshot.digest,
            "dataset": record["legacy"]["dataset"].get("name"),
            "parents": children,
            "lineage_resolution": record["legacy"]["lineage_resolution"],
        }

    return visit(root)


def inventory(store: Store) -> dict[str, Any]:
    """Hash live pairs without retaining bytes or claiming historical consumption."""
    source_revision = _git_revision()
    with store._catalog_lock(exclusive=False):
        items = []
        sidecars = {
            (kind, path.stem): path
            for kind in ("catalog", "ephemeris", "derived")
            for path in (store.processed_dir / kind).glob("*.json")
        }
        parquet = {
            (kind, path.stem): path
            for kind in ("catalog", "ephemeris", "derived")
            for path in (store.processed_dir / kind).glob("*.parquet")
        }
        initial_pins: dict[Path, str] = {}
        for key in sorted(sidecars.keys() | parquet.keys()):
            kind, name = key
            meta_path = sidecars.get(key)
            data_path = parquet.get(key)
            issues = []
            raw = None
            meta_bytes = None
            if meta_path is None:
                issues.append("missing-sidecar")
            else:
                meta_bytes = meta_path.read_bytes()
                initial_pins[meta_path] = digest_bytes(meta_bytes)
                try:
                    raw = _strict_object(meta_bytes, label=str(meta_path))
                except SnapshotError as exc:
                    issues.append(str(exc))
            if data_path is None:
                issues.append("missing-parquet")
            else:
                initial_pins[data_path] = _file_digest(data_path)
            items.append(
                {
                    "kind": kind,
                    "name": name,
                    "legacy_aliases": [f"{kind}/{name}", name],
                    "sidecar_sha256": initial_pins.get(meta_path),
                    "sidecar_bytes_base64": (
                        b64encode(meta_bytes).decode("ascii") if meta_bytes else None
                    ),
                    "parquet_sha256": initial_pins.get(data_path),
                    "metadata": raw,
                    "lineage": raw.get("lineage") if raw else None,
                    "historical_consumption": "unknown",
                    "disposition": "mapped" if not issues else "exception",
                    "exceptions": issues,
                }
            )
        for path, before in initial_pins.items():
            after = (
                digest_bytes(path.read_bytes()) if path.suffix == ".json" else _file_digest(path)
            )
            if after != before:
                raise SourceChangedError(f"source changed during inventory: {path}")
        current_paths = {
            path
            for kind in ("catalog", "ephemeris", "derived")
            for suffix in ("*.json", "*.parquet")
            for path in (store.processed_dir / kind).glob(suffix)
        }
        if current_paths != set(initial_pins):
            raise SourceChangedError("dataset set changed during inventory")
    if _git_revision() != source_revision:
        raise SourceChangedError("source Git revision changed during inventory")
    return {
        "schema_version": 1,
        "kind": "astrolabe-dataset-inventory",
        "dry_run": True,
        "source_revision": source_revision,
        "counts": {
            "sidecars": len(sidecars),
            "parquet": len(parquet),
            "mapped": sum(item["disposition"] == "mapped" for item in items),
            "exceptions": sum(item["disposition"] == "exception" for item in items),
            "lineage_containers": sum(bool(item["lineage"]) for item in items),
        },
        "items": items,
    }


def file_sha256(path: Path | str) -> str:
    """Public helper for migration audits without loading large data files."""
    return _file_digest(Path(path))
