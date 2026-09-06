"""astrolabe CLI.

    astrolabe fetch <source> --name <ds> [--ra --dec --radius | --adql/--sql/--target ...]
    astrolabe query "<sql>"
    astrolabe list
    astrolabe hr --dataset <ds> --out fig.png

Fetch resolves a source adapter, runs its query, and writes the result to the store.
Query runs DuckDB SQL over the processed catalog. Kept thin — the work lives in the
library so it's testable without spawning a process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .sources import SOURCE_NAMES, get_source
from .store import KINDS, Store


def _build_params(args: argparse.Namespace) -> dict[str, Any]:
    """Turn fetch CLI flags into a source params dict."""
    if args.params:
        return json.loads(args.params)
    params: dict[str, Any] = {}
    if args.adql:
        params["adql"] = args.adql
    if args.sql:
        params["sql"] = args.sql
    if args.target:
        params["target"] = args.target
    for key in ("ra", "dec", "radius"):
        val = getattr(args, key)
        if val is not None:
            params[key] = val
    if args.limit is not None:
        params["limit"] = args.limit
    return params


def cmd_fetch(args: argparse.Namespace) -> int:
    source = get_source(args.source)
    params = _build_params(args)
    if not params:
        print("fetch: provide query params (--ra/--dec/--radius, --adql, --sql, "
              "--target, or --params JSON)", file=sys.stderr)
        return 2
    table = source.query(params)
    store = Store(args.data_dir)
    meta = store.write(
        table, name=args.name, source=source.name, kind=source.kind, query=params
    )
    print(f"wrote {meta.n_rows} rows to {meta.kind} dataset {meta.name!r} "
          f"({', '.join(meta.columns)})")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    store = Store(args.data_dir)
    table = store.query(args.sql)
    if args.out:
        store_out = Path(args.out)
        table.write(store_out, overwrite=True)
        print(f"wrote {len(table)} rows to {store_out}")
    else:
        table.pprint_all()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = Store(args.data_dir)
    pairs = store.datasets()
    if not pairs:
        print("(no datasets)")
        return 0
    for kind, name in pairs:
        meta = store.read_meta(name, kind)
        print(f"{kind}/{name}\t{meta.source}\t{meta.n_rows} rows\t{meta.fetched_at}")
    return 0


def cmd_hr(args: argparse.Namespace) -> int:
    from .analysis import hr_diagram

    store = Store(args.data_dir)
    table = store.read(args.dataset)
    fig = hr_diagram(table)
    fig.savefig(args.out, dpi=120)
    print(f"wrote HR diagram to {args.out}")
    return 0


def cmd_snapshot_capture(args: argparse.Namespace) -> int:
    store = Store(args.data_dir)
    snapshot = store.snapshot(
        args.name,
        args.kind,
        parents=args.parent,
        allow_unresolved=args.allow_unresolved,
    )
    print(
        json.dumps(
            {
                "created": snapshot.created,
                "snapshot_digest": snapshot.digest,
                "record": str(snapshot.record_path),
            }
        )
    )
    return 0


def cmd_snapshot_read(args: argparse.Namespace) -> int:
    store = Store(args.data_dir)
    table = store.read_snapshot(args.digest)
    if args.out:
        output = Path(args.out)
        table.write(output, overwrite=False)
        print(json.dumps({"rows": len(table), "output": str(output)}))
    else:
        table.pprint_all()
    return 0


def cmd_snapshot_restore(args: argparse.Namespace) -> int:
    store = Store(args.data_dir)
    meta = store.restore_snapshot(args.digest, overwrite=args.overwrite)
    print(json.dumps({"restored": f"{meta.kind}/{meta.name}", "rows": meta.n_rows}))
    return 0


def cmd_snapshot_trace(args: argparse.Namespace) -> int:
    from .provenance import trace_snapshot

    print(json.dumps(trace_snapshot(Store(args.data_dir), args.digest), indent=2))
    return 0


def cmd_snapshot_inventory(args: argparse.Namespace) -> int:
    from .provenance import inventory

    result = inventory(Store(args.data_dir))
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(json.dumps({"output": str(output), "counts": result["counts"]}))
    else:
        print(encoded, end="")
    return 0


def cmd_snapshot_manifest(args: argparse.Namespace) -> int:
    from .provenance import export_manifest

    output = export_manifest(args.record, args.output)
    print(json.dumps({"output": str(output), "records": len(args.record)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrolabe", description=__doc__)
    parser.add_argument(
        "--data-dir", default="data", help="catalog root (default: data)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="fetch from a source into the catalog")
    p_fetch.add_argument("source", choices=SOURCE_NAMES)
    p_fetch.add_argument("--name", required=True, help="dataset name to store under")
    p_fetch.add_argument("--ra", type=float)
    p_fetch.add_argument("--dec", type=float)
    p_fetch.add_argument("--radius", type=float, help="cone radius (deg)")
    p_fetch.add_argument("--limit", type=int)
    p_fetch.add_argument("--adql", help="Gaia ADQL passthrough")
    p_fetch.add_argument("--sql", help="SDSS SQL passthrough")
    p_fetch.add_argument("--target", help="Horizons target id")
    p_fetch.add_argument("--params", help="raw JSON params dict (overrides flags)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_query = sub.add_parser("query", help="run DuckDB SQL over the catalog")
    p_query.add_argument("sql")
    p_query.add_argument("--out", help="write result to a Parquet/ECSV path")
    p_query.set_defaults(func=cmd_query)

    p_list = sub.add_parser("list", help="list stored datasets")
    p_list.set_defaults(func=cmd_list)

    p_hr = sub.add_parser("hr", help="render an HR diagram from a dataset")
    p_hr.add_argument("--dataset", required=True)
    p_hr.add_argument("--out", default="hr.png")
    p_hr.set_defaults(func=cmd_hr)

    p_snapshot = sub.add_parser(
        "snapshot", help="capture and use immutable orbit-research dataset pins"
    )
    snapshot_sub = p_snapshot.add_subparsers(dest="snapshot_command", required=True)

    p_capture = snapshot_sub.add_parser(
        "capture", help="retain one mutable dataset generation"
    )
    p_capture.add_argument("name")
    p_capture.add_argument("--kind", choices=KINDS)
    p_capture.add_argument(
        "--parent",
        action="append",
        default=[],
        help="exact parent snapshot record.json; repeat for each lineage parent",
    )
    p_capture.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="retain explicit pending legacy lineage (not reproducible-run export)",
    )
    p_capture.set_defaults(func=cmd_snapshot_capture)

    p_read = snapshot_sub.add_parser("read", help="verify and read retained bytes")
    p_read.add_argument("digest")
    p_read.add_argument("--out")
    p_read.set_defaults(func=cmd_snapshot_read)

    p_restore = snapshot_sub.add_parser(
        "restore", help="restore retained bytes to their mutable alias"
    )
    p_restore.add_argument("digest")
    p_restore.add_argument("--overwrite", action="store_true")
    p_restore.set_defaults(func=cmd_snapshot_restore)

    p_trace = snapshot_sub.add_parser("trace", help="trace exact local parent pins")
    p_trace.add_argument("digest")
    p_trace.set_defaults(func=cmd_snapshot_trace)

    p_inventory = snapshot_sub.add_parser(
        "inventory", help="dry-run inventory without retaining or changing bytes"
    )
    p_inventory.add_argument("--output")
    p_inventory.set_defaults(func=cmd_snapshot_inventory)

    p_manifest = snapshot_sub.add_parser(
        "manifest", help="export an exact v1 manifest from snapshot records"
    )
    p_manifest.add_argument("--record", action="append", required=True)
    p_manifest.add_argument("--output", required=True)
    p_manifest.set_defaults(func=cmd_snapshot_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
