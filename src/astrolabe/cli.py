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
from .store import Store


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
