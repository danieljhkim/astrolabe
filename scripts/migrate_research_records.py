#!/usr/bin/env python3
"""Build/check the ORB-11380 owner migration from a read-only inventory.

This is an Astrolabe mapping adapter, not a second record engine. Archived records,
references, revision hashes, and manifests retain their orbit-research v1 contract;
the exact-pinned v0.2 package validates that compatibility surface. It never reads
tables, runs analyses, changes source data, or puts Parquet in Git.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from base64 import b64decode
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from orbit_research import make_record, validate
from orbit_research.contract import reference, revision_digest

BASELINE = "90f5b58890da36c44286a4edbde7eead879410a8"
FRAMEWORK = "0a9cf756e1c2522b9d5ee71c1cf462b8676f4281"
HISTORICAL_FRAMEWORK = "7b6c1b2380bc915d6ff7cca50f288ed716a99c74"
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "research" / "datasets"
EXPECTED_DATASETS = 27
WIDEBIN_DATASETS = (
    ("catalog", "widebin_gaia_dr3_d200"),
    ("catalog", "widebin_pairs_baseline"),
    ("derived", "widebin_vtilde_gn"),
    ("derived", "widebin_sensitivity"),
)
PRINCIPIA_FREEZE = "c04f2ed1ae91d6c126bc60863b5e48f46abe4576"
PRINCIPIA_REFS = (
    (
        "urn:research:principia:protocol:wide-binary-selection-bias-control",
        "sha256:90b62ef438d73d1a9ceb25c4d7753c0d4fce49dbacbcc207c7febc17f7c70842",
    ),
    (
        "urn:research:principia:program:wide-binary-selection-methodology",
        "sha256:8984ee8cdbd455526503d40ce5f867b42d33e5a27428c4a630892d1bd22552df",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def package_check() -> None:
    dist = distribution("orbit-research")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    require(
        dist.version == "0.2.0" and direct.get("vcs_info", {}).get("commit_id") == FRAMEWORK,
        "install requirements-research.txt: exact orbit-research revision required",
    )


def provenance(
    path: str,
    sha: str,
    *,
    blob_oid: str | None = None,
    selector: str = "$",
    historical: bool = True,
    working_tree: bool = True,
) -> dict[str, Any]:
    return {
        "repository": "astrolabe",
        "git_revision": BASELINE,
        "blob_oid": blob_oid,
        "sha256": sha,
        "path": path,
        "selector": selector,
        "historical": historical,
        "working_tree": working_tree,
    }


def record_path(record: dict[str, Any]) -> str:
    return f"records/{record['kind']}-{record['revision_id'].split(':')[1]}.json"


def dataset_record(item: dict[str, Any]) -> dict[str, Any]:
    kind, name = item["kind"], item["name"]
    sidecar_bytes = b64decode(item["sidecar_bytes_base64"], validate=True)
    require(digest(sidecar_bytes) == item["sidecar_sha256"], "sidecar byte pin changed")
    require(json.loads(sidecar_bytes) == item["metadata"], "sidecar metadata differs from bytes")
    require(item["disposition"] == "mapped" and not item["exceptions"], "unmapped dataset")
    lineage = item["lineage"]
    resolutions = []
    for index, entry in enumerate(lineage or []):
        dataset = entry.get("dataset") if isinstance(entry, dict) else None
        resolutions.append(
            {
                "index": index,
                "legacy": entry,
                "status": "pending",
                "reason": (
                    "Legacy parent was never persisted."
                    if dataset is None
                    else "Legacy name/timestamp is not an immutable historical parent pin."
                ),
            }
        )
    missing = ["immutable-retained-bytes", "historical-consumption", "scope"]
    if lineage:
        missing.append("historical-lineage-unresolved")
    elif lineage is None:
        missing.append("lineage-not-recorded")
    record = make_record(
        "astrolabe",
        "artifact",
        f"dataset:{kind}:{name}",
        {
            "role": "dataset",
            "availability": "external",
            "snapshot_digest": item["parquet_sha256"],
            "locator": f"data/processed/{kind}/{name}.parquet",
            "media_type": "application/vnd.apache.parquet",
        },
        provenance(f"data/processed/{kind}/{name}.json", item["sidecar_sha256"]),
        legacy={
            "dataset": item["metadata"],
            "sidecar_bytes_base64": item["sidecar_bytes_base64"],
            "inventory_pins": {
                "sidecar_sha256": item["sidecar_sha256"],
                "parquet_sha256": item["parquet_sha256"],
            },
            "lineage_resolution": resolutions,
        },
        activity="active",
        scope="unknown",
        limitations=[
            "Dry-run current-byte hash; retained immutable bytes require snapshot capture.",
            "Present bytes do not establish which generation any historical analysis consumed.",
        ],
        missingness=missing,
    )
    record["aliases"] = [f"{kind}/{name}", name]
    record["revision_id"] = revision_digest(record)
    require(not validate(record), f"invalid dataset record for {kind}/{name}")
    return record


def git_source_record(path: str) -> dict[str, Any]:
    data = git("show", f"{BASELINE}:{path}")
    blob = git("rev-parse", f"{BASELINE}:{path}").decode().strip()
    record = make_record(
        "astrolabe",
        "artifact",
        f"source:astrolabe:{BASELINE}:{path}",
        {
            "role": "source",
            "availability": "available",
            "snapshot_digest": digest(data),
            "locator": path,
            "media_type": "text/x-python",
        },
        provenance(path, digest(data), blob_oid=blob, working_tree=False),
        legacy={"source_revision": BASELINE, "bytes_sha256": digest(data)},
        activity="active",
        scope="synthetic-calibration",
        limitations=[
            "Source-code artifact only; execution and scientific assessment are separate."
        ],
    )
    require(not validate(record), f"invalid source record for {path}")
    return record


def missing_dataset_record(kind: str, name: str, source: dict[str, Any]) -> dict[str, Any]:
    record = make_record(
        "astrolabe",
        "artifact",
        f"dataset:{kind}:{name}",
        {
            "role": "dataset",
            "availability": "missing",
            "snapshot_digest": None,
            "locator": f"data/processed/{kind}/{name}.parquet",
            "media_type": "application/vnd.apache.parquet",
        },
        provenance(
            "scripts/run_wide_binary_study.py",
            source["provenance"]["sha256"],
            blob_oid=source["provenance"]["blob_oid"],
            selector=f"dataset:{kind}/{name}",
            working_tree=False,
        ),
        legacy={
            "expected_dataset": {"kind": kind, "name": name},
            "availability_at_inventory": "missing",
            "historical_consumption": "unknown",
        },
        activity="unknown",
        scope="unknown",
        limitations=[
            "Expected study output only; no dataset bytes or historical consumption were inferred."
        ],
        missingness=["dataset-snapshot", "historical-consumption", "scope", "activity"],
    )
    record["aliases"] = [f"{kind}/{name}", name]
    record["revision_id"] = revision_digest(record)
    require(not validate(record), f"invalid missing dataset record for {kind}/{name}")
    return record


def framework_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "kind": "manifest",
        "repositories": [{"id": "astrolabe", "git_revision": BASELINE}],
        "references": [reference(record) for record in records],
    }
    require(not validate(manifest), "invalid dataset owner manifest")
    return manifest


def build(inventory: dict[str, Any]) -> dict[str, bytes]:
    counts = inventory.get("counts", {})
    require(inventory.get("dry_run") is True, "migration requires a dry-run inventory")
    require(inventory.get("source_revision") == BASELINE, "inventory source revision changed")
    require(
        counts
        == {
            "sidecars": EXPECTED_DATASETS,
            "parquet": EXPECTED_DATASETS,
            "mapped": EXPECTED_DATASETS,
            "exceptions": 0,
            "lineage_containers": 11,
        },
        f"live inventory differs from reviewed 27-pair baseline: {counts}",
    )
    records = [dataset_record(item) for item in inventory["items"]]
    require(
        len({record["id"] for record in records}) == EXPECTED_DATASETS, "dataset identity collision"
    )
    apparatus = git_source_record("src/astrolabe/analysis/wide_binaries.py")
    script_source = git_source_record("scripts/run_wide_binary_study.py")
    missing = [missing_dataset_record(kind, name, script_source) for kind, name in WIDEBIN_DATASETS]

    outputs = {
        record_path(record): encoded(record)
        for record in records + [apparatus, script_source] + missing
    }
    dataset_manifest = framework_manifest(records)
    outputs["dataset-manifest.json"] = encoded(dataset_manifest)

    chain_refs = [reference(apparatus), reference(script_source), *map(reference, missing)]
    chain_refs.extend(
        {
            "repository": "principia",
            "id": ident,
            "revision_id": revision,
            "source_revision": PRINCIPIA_FREEZE,
            "status": "pending",
        }
        for ident, revision in PRINCIPIA_REFS
    )
    chain_manifest = {
        "schema_version": 1,
        "kind": "manifest",
        "repositories": [
            {"id": "astrolabe", "git_revision": BASELINE},
            {"id": "principia", "git_revision": PRINCIPIA_FREEZE},
        ],
        "references": chain_refs,
    }
    require(not validate(chain_manifest), "invalid wide-binary chain manifest")
    outputs["wide-binary-chain-manifest.json"] = encoded(chain_manifest)

    output_records = records + [apparatus, script_source] + missing
    migration = {
        "schema_version": 1,
        "migration": "ORB-11380",
        "authority": "Astrolabe-owned orbit-research v1 artifact records",
        "framework": {
            "version": "0.1.0",
            "contract": 1,
            "git_revision": HISTORICAL_FRAMEWORK,
        },
        "source_revision": BASELINE,
        "inventory": {
            "path": "inventory.json",
            "sha256": digest(encoded(inventory)),
            "reviewed_counts": counts,
            "difference_from_foundation": "none: 27 sidecars and 27 Parquet files",
        },
        "records": [
            {
                "path": record_path(record),
                "id": record["id"],
                "revision_id": record["revision_id"],
                "sha256": digest(outputs[record_path(record)]),
            }
            for record in output_records
        ],
        "aliases": [
            {"alias": alias, "id": record["id"], "revision_id": record["revision_id"]}
            for record in output_records
            for alias in record["aliases"]
        ],
        "lineage": [
            {
                "source": reference(record),
                "entries": record["legacy"]["lineage_resolution"],
                "status": "pending",
                "reason": "Current hashing cannot prove historical parent consumption.",
            }
            for record in records
            if record["legacy"]["lineage_resolution"]
        ],
        "framework_manifests": [
            "dataset-manifest.json",
            "wide-binary-chain-manifest.json",
        ],
        "pending_reconciliation": [
            {
                "repository": "orrery",
                "source_revision": revision,
                "path": f"lab/sims/{slug}/sim.json",
                "id": None,
                "revision_id": None,
                "status": "pending",
                "reason": (
                    "Orrery has not published an exact canonical record tuple; none is invented."
                ),
            }
            for revision, slug in (
                ("28dd5c72bb670517b93b556f1d2483402c8e8655", "wide-binary-selection-bias"),
                ("2e097e606bc751ba1a8b29ebdc5aab6bbd961c43", "wide-binary-control-diagnosis"),
            )
        ],
        "exceptions": [
            {
                "id": "current-digest-not-historical-consumption",
                "count": EXPECTED_DATASETS,
                "reason": (
                    "All present-day hashes are mapped, but no old run is assigned "
                    "those bytes without evidence."
                ),
            },
            {
                "id": "legacy-lineage-not-exact-pins",
                "count": 11,
                "reason": (
                    "All legacy lineage containers and entries are preserved "
                    "pending exact owner pins."
                ),
            },
            {
                "id": "never-persisted-gaia-parent",
                "count": 1,
                "reason": (
                    "field_eq180_xmatch retains its null Gaia lineage parent; no "
                    "current dataset is substituted."
                ),
            },
            {
                "id": "v1-non-git-artifact-resolution-gap",
                "reason": (
                    "v1 reconciliation requires clean Git provenance for resolved "
                    "references; owner-local content-addressed datasets remain exact "
                    "pending pins."
                ),
            },
        ],
        "rollback": {
            "command": (
                "python3 scripts/migrate_research_records.py rollback --output <new-directory>"
            ),
            "scope": (
                "exact legacy sidecar bytes only; immutable snapshot directories "
                "and new evidence are retained"
            ),
        },
    }
    outputs["migration.json"] = encoded(migration)
    outputs["inventory.json"] = encoded(inventory)
    return outputs


def write_new_tree(outputs: dict[str, bytes], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        for relative, data in outputs.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
    except BaseException:
        shutil.rmtree(destination)
        raise


def files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def check() -> None:
    inventory = json.loads((AUTHORITY / "inventory.json").read_bytes())
    expected = build(inventory)
    actual = files(AUTHORITY)
    require(actual == expected, "tracked research/datasets differs from deterministic migration")
    for relative, data in actual.items():
        document = json.loads(data)
        if document.get("kind") in {"artifact", "manifest"}:
            require(not validate(document), f"orbit-research validation failed: {relative}")


def rollback(destination: Path) -> None:
    if destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("rollback output must be outside the checkout")
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    inventory = json.loads((AUTHORITY / "inventory.json").read_bytes())
    destination.mkdir(parents=True)
    try:
        for item in inventory["items"]:
            path = destination / "processed" / item["kind"] / f"{item['name']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b64decode(item["sidecar_bytes_base64"], validate=True))
            require(digest(path.read_bytes()) == item["sidecar_sha256"], "rollback byte mismatch")
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--inventory", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    commands.add_parser("check")
    rollback_parser = commands.add_parser("rollback")
    rollback_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    package_check()
    if args.command == "build":
        write_new_tree(build(json.loads(args.inventory.read_bytes())), args.output)
    elif args.command == "check":
        check()
    else:
        rollback(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
