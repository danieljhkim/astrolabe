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
- **Store**: raw fetches land in `data/raw/<source>/`; normalized tables live under
  `data/processed/<kind>/` (kind ∈ `catalog` | `ephemeris` | `derived`) as Parquet with a
  sidecar JSON (kind, source, query, fetched_at, row count; derived datasets carry
  `lineage` naming their parents). Names follow a per-kind grammar and must be
  deterministic w.r.t. their query — overwrite-by-name is a refresh, a changed query is a
  new name (full layout + grammar in `store.py`'s docstring). `data/scratch/` is free-form
  notebook space outside the catalog. `data/` is gitignored.
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
- [x] **M5 — ephemeris (stretch)**: JPL Horizons adapter (`sources/horizons.py`,
      normalized to ra/dec + target `source_id`). Storage resolved (2026-07): ephemerides
      share `store.py` but land under their own `data/processed/ephemeris/` kind (see §7).
- [x] **M6 — external galaxies (SPARC, ORB-10168)**: `sources/sparc.py` over the
      published Lelli+2016 MRT tables (whole-survey download, no ra/dec — SPARC
      publishes none), `analysis/rar.py` radial-acceleration-relation arrays, sidecar
      `semantics` field for cross-repo deliveries, `scripts/deliver_sparc.py` +
      `docs/sparc-interface.md` (the ws_orrery universality-fit contract).
- [x] **M7 — wide binaries (ORB-10753)**: `sources/wide_binaries.py` (Gaia DR3
      local-volume ADQL), `analysis/wide_binaries.py` (El-Badry-style pairing,
      chance-alignment, ṽ(g_N) vs Newtonian mocks, sensitivity table),
      `scripts/run_wide_binary_study.py` + notebook. Predeclared cuts/ecc prior
      before binning; honest Newtonian comparison (not circular-orbit analytic).

## 6. Constellation integration

- Register in `operations/scripts/repos.tsv`; branch `agent-main`.
- Own thin `CLAUDE.md` (deep context here, not in the umbrella router).
- Orbit workspace (`ws_astrolabe`) only once M2 lands and there's dispatchable work;
  structured decisions (e.g., storage format changes) recorded as ADRs in polaris.

## 7. Open questions

- Column-naming convention for normalized tables (adopt Gaia's? define a minimal house set?)
- ~~Dataset versioning~~ — **resolved 2026-07**: overwrite-by-name; the sidecar's exact
  query makes datasets reproducible, and a name must be deterministic w.r.t. its query
  (time-dependent identity goes in the name, e.g. `mars_2026`). See `store.py` docstring.
- ~~Ephemeris/time-series storage path~~ — **resolved 2026-07**: shares `store.py`, but
  every dataset is kind-partitioned under `data/processed/<kind>/` (`catalog` |
  `ephemeris` | `derived`); adapters declare their kind (`Source.kind`), and derived
  datasets record `lineage`. Decided with Daniel in-session; recorded here (no ADR).
