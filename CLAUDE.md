# astrolabe

Celestial data collection & analysis (Python). Local library + CLI — **no web service,
no APIs** (SPEC §2). Independent repo under `codebases/astrolabe`, branch `agent-main`,
direct commits (not PR-gated).

[SPEC.md](SPEC.md) is the source of truth for design + milestones; keep its checklist
current as work lands.

## Orientation

- **`src/astrolabe/sources/`** — one adapter per data source; each implements the
  `Source` protocol in [base.py](src/astrolabe/sources/base.py) (`name` + `kind` +
  `query(params) -> astropy.table.Table`). **The load-bearing seam: nothing outside
  `sources/` may import a provider SDK.** Adapters normalize to the house standard
  columns (`source_id`, `ra`, `dec`); provider imports and the network call are
  isolated in a small method (`_run*`) so tests inject fixtures instead of hitting the
  network.
- **`store.py`** — catalog: astropy Table ↔ Parquet under `data/` (gitignored), DuckDB
  query layer. Datasets are kind-partitioned: `data/processed/<kind>/<name>.parquet` +
  sidecar `<name>.json`, kind ∈ `catalog` | `ephemeris` | `derived` (adapters declare
  theirs via `Source.kind`; derived datasets carry `lineage` naming their parents).
  Names follow a per-kind grammar, deterministic w.r.t. the query — layout, grammar, and
  the overwrite rule live in `store.py`'s module docstring. Raw fetches in
  `data/raw/<source>/`; `data/scratch/` is free-form notebook space outside the catalog.
  `Store.query` registers views as `<kind>.<name>` (plus an unqualified alias while a
  name is unique across kinds).
- **`analysis/`** — pure functions (Table in, Table/Figure out): `crossmatch`,
  `hr_diagram`. No network, no I/O side effects.
- **`cli.py`** — `astrolabe fetch|query|list|hr`.

## Working here

- `uv sync --extra dev`, then `uv run pytest` (offline — fixtures/mocks, no network) and
  `uv run ruff check .`.
- **Adding a source**: implement `Source`, normalize to standard columns, keep the
  provider import inside the adapter's `_run*` seam, register the slug in
  [sources/__init__.py](src/astrolabe/sources/__init__.py).
- Notebooks in `notebooks/` for exploration; promote stable code into `src/` (SPEC §4).

## Constellation

- Structured decisions (storage format, column convention — see SPEC §7) → ADRs in
  polaris, not inline here.
- Orbit workspace **`ws_astrolabe`** is provisioned (dk-server-1, ship-mode `local`,
  base branch `agent-main`) — research tasks are tracked there (SPEC §6).
- Research work here is performed by **tycho**, the astrophysicist agent
  (`agentbase/tycho/memory`).
