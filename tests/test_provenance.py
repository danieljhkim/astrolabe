"""Offline immutable snapshot and orbit-research v1 integration tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from base64 import b64decode
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from orbit_research import validate

from astrolabe.provenance import (
    SnapshotConflictError,
    SnapshotError,
    SourceChangedError,
    export_manifest,
    inventory,
    trace_snapshot,
)
from astrolabe.store import Store


def test_snapshot_survives_overwrite_and_is_idempotent(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia", query={"release": "DR3"})

    first = store.snapshot("stars")
    repeated = store.snapshot("stars")
    assert first.created is True
    assert repeated.created is False
    assert repeated.digest == first.digest
    assert validate(first.record) == []
    assert first.record["legacy"]["snapshot"]["units"] == {
        "bp_rp": "mag",
        "dec": "deg",
        "parallax": "mas",
        "phot_g_mean_mag": "mag",
        "ra": "deg",
    }
    assert [field["name"] for field in first.record["legacy"]["snapshot"]["schema"]] == list(
        gaia_table.colnames
    )

    replacement = gaia_table.copy()
    replacement["ra"] = replacement["ra"] + 12
    store.write(replacement, name="stars", source="gaia", query={"release": "DR3"})
    current = store.read("stars")
    retained = store.read_snapshot(first.digest)
    assert not np.allclose(current["ra"], retained["ra"])
    np.testing.assert_allclose(retained["ra"], gaia_table["ra"])


def test_snapshot_restore_is_exact_and_guarded(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    snapshot = store.snapshot("stars")
    exact_data = snapshot.data_path.read_bytes()
    exact_meta = snapshot.metadata_path.read_bytes()

    with pytest.raises(FileExistsError):
        store.restore_snapshot(snapshot.digest)
    store.write(gaia_table[:1], name="stars", source="gaia")
    restored = store.restore_snapshot(snapshot.digest, overwrite=True)
    assert restored.n_rows == len(gaia_table)
    assert store._processed_path("catalog", "stars").read_bytes() == exact_data
    assert store._meta_path("catalog", "stars").read_bytes() == exact_meta


def test_derived_snapshot_requires_exact_parent_and_traces_it(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    parent = store.snapshot("stars")
    store.write(
        gaia_table[:2],
        name="pairs",
        source="analysis.select",
        kind="derived",
        lineage=[{"dataset": "stars", "fetched_at": "2026-01-01T00:00:00+00:00"}],
    )
    with pytest.raises(SnapshotError, match="unresolved parents"):
        store.snapshot("pairs", "derived")
    child = store.snapshot("pairs", "derived", parents=[parent.record_path])
    assert child.record["references"][0]["revision_id"] == parent.record["revision_id"]
    trace = trace_snapshot(store, child.digest)
    assert trace["parents"][0]["dataset"] == "stars"


def test_missing_historical_parent_stays_explicit_and_cannot_export(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(
        gaia_table,
        name="xmatch",
        source="analysis.crossmatch",
        kind="derived",
        lineage=[{"dataset": None, "note": "Gaia input was never persisted"}],
    )
    snapshot = store.snapshot("xmatch", "derived", allow_unresolved=True)
    resolution = snapshot.record["legacy"]["lineage_resolution"][0]
    assert resolution["status"] == "pending"
    assert "never persisted" in resolution["reason"]
    with pytest.raises(SnapshotError, match="cannot be exported"):
        export_manifest([snapshot.record_path], tmp_path / "manifest.json")


def test_changed_parent_is_rejected(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    parent = store.snapshot("stars")
    parent.data_path.write_bytes(b"changed")
    store.write(
        gaia_table,
        name="derived",
        source="analysis.test",
        kind="derived",
        lineage=[{"dataset": "stars"}],
    )
    with pytest.raises(SnapshotConflictError, match="Parquet bytes"):
        store.snapshot("derived", "derived", parents=[parent.record_path])


def test_uncoordinated_source_change_fails_closed(tmp_path, gaia_table, monkeypatch):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    source = store._processed_path("catalog", "stars")
    real_copyfile = shutil.copyfile

    def changing_copy(src, dst, *args, **kwargs):
        result = real_copyfile(src, dst, *args, **kwargs)
        if src == source:
            with source.open("ab") as stream:
                stream.write(b"changed concurrently")
        return result

    monkeypatch.setattr(shutil, "copyfile", changing_copy)
    with pytest.raises(SourceChangedError, match="changed during snapshot"):
        store.snapshot("stars")
    assert list(store.snapshot_dir.glob("*")) == []


def test_partial_capture_and_conflicting_destination_are_safe(tmp_path, gaia_table, monkeypatch):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    import astrolabe.provenance as provenance

    real_rename = provenance.os.rename
    monkeypatch.setattr(
        provenance.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected rename failure")),
    )
    with pytest.raises(OSError, match="injected"):
        store.snapshot("stars")
    assert list(store.snapshot_dir.glob("*")) == []

    monkeypatch.setattr(provenance.os, "rename", real_rename)
    snapshot = store.snapshot("stars")
    snapshot.record_path.write_text("{}")
    with pytest.raises(SnapshotConflictError, match="conflicts"):
        store.snapshot("stars")


def test_manifest_and_inventory_validate_without_reading_science(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    snapshot = store.snapshot("stars")
    output = export_manifest([snapshot.record_path], tmp_path / "manifest.json")
    manifest = json.loads(output.read_text())
    assert validate(manifest) == []
    assert manifest["references"][0]["status"] == "pending"
    report = inventory(store)
    assert report["counts"] == {
        "sidecars": 1,
        "parquet": 1,
        "mapped": 1,
        "exceptions": 0,
        "lineage_containers": 0,
    }
    assert report["items"][0]["historical_consumption"] == "unknown"


def test_legacy_sidecar_without_integrity_fields_remains_capturable(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    meta_path = store._meta_path("catalog", "stars")
    raw = json.loads(meta_path.read_text())
    for field in ("parquet_sha256", "schema", "units"):
        raw.pop(field)
    meta_path.write_text(json.dumps(raw))
    snapshot = store.snapshot("stars")
    assert snapshot.record["legacy"]["dataset"] == raw
    assert snapshot.record["missingness"] == [
        "non-git-dataset-bytes",
        "lineage-not-recorded",
        "scope",
    ]


def test_read_snapshot_refuses_tampered_metadata(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    snapshot = store.snapshot("stars")
    snapshot.metadata_path.write_text("{}")
    with pytest.raises(SnapshotConflictError, match="sidecar bytes"):
        store.read_snapshot(snapshot.digest)


def test_snapshot_roundtrip_preserves_table_type(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    result = store.read_snapshot(store.snapshot("stars").digest)
    assert isinstance(result, Table)


def test_tracked_migration_is_complete_valid_and_rollback_exact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    authority = root / "research" / "datasets"
    subprocess.run(
        [sys.executable, "scripts/migrate_research_records.py", "check"],
        cwd=root,
        check=True,
    )
    migration = json.loads((authority / "migration.json").read_text())
    inventory_report = json.loads((authority / "inventory.json").read_text())
    assert migration["inventory"]["reviewed_counts"] == {
        "sidecars": 27,
        "parquet": 27,
        "mapped": 27,
        "exceptions": 0,
        "lineage_containers": 11,
    }
    records = [json.loads(path.read_text()) for path in (authority / "records").glob("*.json")]
    assert len([r for r in records if r["legacy"].get("dataset")]) == 27
    assert all(validate(record) == [] for record in records)
    for manifest_name in (
        "dataset-manifest.json",
        "wide-binary-chain-manifest.json",
    ):
        assert validate(json.loads((authority / manifest_name).read_text())) == []

    rollback = tmp_path / "legacy-sidecars"
    subprocess.run(
        [
            sys.executable,
            "scripts/migrate_research_records.py",
            "rollback",
            "--output",
            str(rollback),
        ],
        cwd=root,
        check=True,
    )
    for item in inventory_report["items"]:
        path = rollback / "processed" / item["kind"] / f"{item['name']}.json"
        assert path.read_bytes() == b64decode(item["sidecar_bytes_base64"])
