# astrolabe — SPEC

Celestial data collection & analysis. Python. Lives at `codebases/astrolabe`; independent repo
on `agent-main` (direct commits, not PR-gated). This spec is the source of truth for build-out;
implementing agents should update the checklist below as milestones land.

## 1. Purpose & scope

- Collect celestial data from public surveys/APIs (Gaia, SDSS, JPL Horizons, SIMBAD, VizieR)
  into a local catalog, and analyze it (cross-matching, HR diagrams, light curves).
- Data sources are deliberately open-ended: telescope captures (FITS) and ephemeris/event
  tracking are anticipated later — the design must absorb them as new adapters only.

## 2. Non-goals

- No web service, no exposed APIs — local library + CLI only.
- No real-time pipelines; batch fetch/analyze.
- No custom search/indexing — DuckDB queries over Parquet are sufficient. Sextant integration,
  if ever, is a later ADR.

## 3. Architecture

```
src/astrolabe/
├── sources/      # one adapter per data source; each implements the Source protocol
│   ├── base.py   # Source protocol: name, query(params) -> astropy.table.Table
│   ├── gaia.py   # via astroquery.gaia (TAP/ADQL)
│   └── sdss.py   # via astroquery.sdss
├── store.py      # catalog: astropy Table <-> Parquet under data/; DuckDB query layer
├── analysis/     # pure functions over Tables/DataFrames (crossmatch, hr_diagram, ...)
└── cli.py        # `astrolabe fetch <source> ...`, `astrolabe query <sql>`
```

- **Source protocol** is the load-bearing abstraction: `query() -> astropy Table` with
  standardized units/columns per astropy conventions. Adapters are isolated; nothing outside
  `sources/` may import a specific provider SDK.
- **Store**: raw fetches land in `data/raw/<source>/`, normalized tables in
  `data/processed/` as Parquet with dataset metadata (source, query, fetched_at, row count)
  in a sidecar JSON. `data/` is gitignored.
- **Analysis** functions are pure (Table in, Table/figure out) and unit-tested against small
  fixture tables — no network in tests.

## 4. Stack & tooling

- Python ≥3.12, `uv` for env/deps. Core deps: astropy, astroquery, duckdb, pyarrow,
  matplotlib. Dev: pytest, ruff.
- Notebooks under `notebooks/` for exploration; stable code must be promoted into `src/`.

## 5. Milestones

- [x] **M1 — skeleton**: repo scaffold, pyproject, Source protocol, store round-trip
      (Table -> Parquet -> DuckDB), pytest green.
- [x] **M2 — first source**: Gaia adapter (cone search + ADQL passthrough), `fetch` CLI,
      fixture-based tests.
- [x] **M3 — analysis v1**: cross-match two catalogs, HR diagram from Gaia photometry.
- [x] **M4 — second source (SDSS)**: proves the adapter seam; protocol held (only the
      `objid -> source_id` rename differed — absorbed by per-adapter normalization).
- [~] **M5 — ephemeris (stretch)**: JPL Horizons adapter scaffolded (`sources/horizons.py`,
      normalized to ra/dec + target `source_id`). Time-series storage decision still open —
      deferred to an ADR (see §7); ephemerides currently flow through the same `store.py`.

## 6. Constellation integration

- Register in `operations/scripts/repos.tsv`; branch `agent-main`.
- Own thin `CLAUDE.md` (deep context here, not in the umbrella router).
- Orbit workspace (`ws_astrolabe`) only once M2 lands and there's dispatchable work;
  structured decisions (e.g., storage format changes) recorded as ADRs in polaris.

## 7. Open questions

- Column-naming convention for normalized tables (adopt Gaia's? define a minimal house set?)
- Dataset versioning: overwrite vs. append-with-fetched_at partitions in `data/processed/`.
- Whether ephemeris/event data (time-series) shares `store.py` or gets its own path (M5).
