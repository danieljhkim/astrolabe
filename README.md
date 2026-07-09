# astrolabe

Celestial data collection & analysis. A local library + CLI that fetches from public
surveys (Gaia, SDSS, …) into a Parquet catalog, queries it with DuckDB, and runs pure
analysis functions (cross-match, HR diagram) over astropy Tables.

No web service, no APIs — local only. See [SPEC.md](SPEC.md) for the full design and
milestones.

## Quickstart

```sh
uv sync --extra dev
uv run pytest            # tests are offline (fixtures/mocks, no network)

# fetch a Gaia cone search into the catalog (requires network)
uv run astrolabe fetch gaia --name m31_core --ra 10.68 --dec 41.27 --radius 0.05

# query the catalog with DuckDB SQL
uv run astrolabe query "SELECT source_id, ra, dec FROM m31_core WHERE dec > 41.2"

# list datasets / render an HR diagram
uv run astrolabe list
uv run astrolabe hr --dataset m31_core --out m31_hr.png
```

## Layout

```
src/astrolabe/
├── sources/      # one adapter per source, each implements the Source protocol
│   ├── base.py   # Source protocol + standard-column normalization
│   ├── gaia.py   # astroquery.gaia (TAP/ADQL)
│   ├── sdss.py   # astroquery.sdss
│   └── horizons.py  # JPL Horizons ephemerides (stretch)
├── store.py      # Table <-> Parquet under data/, DuckDB query layer
├── analysis/     # pure functions: crossmatch, hr_diagram
└── cli.py        # astrolabe fetch|query|list|hr
```

Fetched data lands under `data/` (gitignored): raw as-fetched in `data/raw/<source>/`,
normalized in `data/processed/` as Parquet with a sidecar `<name>.json` (source, query,
fetched_at, row count).

## Adding a source

Implement the `Source` protocol (`name` + `query(params) -> astropy.table.Table`),
normalize to the standard columns (`source_id`, `ra`, `dec`), isolate the provider SDK
import inside the adapter, and register the slug in `sources/__init__.py`. Nothing
outside `sources/` may import a provider SDK.
