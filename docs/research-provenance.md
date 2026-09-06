# Immutable dataset provenance

Astrolabe uses **orbit-research 0.1.0, contract v1**, pinned to Git revision
`7b6c1b2380bc915d6ff7cca50f288ed716a99c74`. The framework owns record identity,
revision hashing, references, manifests, and validation. Astrolabe supplies only the
dataset-specific capture and legacy mapping. Orbit remains the operational task/run
authority; neither package adds a service, scheduler, or shadow task store.

## Install the exact contract

The normal development extra includes the reviewed framework revision:

```sh
UV_CACHE_DIR=/tmp/astrolabe-uv-cache uv sync --extra dev
```

For a research-only environment, install `requirements-research.txt`. The direct Git
revision is intentional: an unrelated package that happens to call itself version 0.1.0
is not an equivalent contract.

## Mutable aliases and immutable snapshots

`Store.write` and existing `read`, `query`, and `list` behavior remain compatible.
Mutable aliases still live at `data/processed/<kind>/<name>.{parquet,json}`. New writes
publish the pair under a catalog lock, put the exact Parquet SHA-256, table schema, and
units in the sidecar, and roll back to the previous pair if publication fails.

Scientific consumption captures a generation first:

```sh
uv run astrolabe --data-dir data snapshot capture widebin_gaia_dr3_d200 --kind catalog
```

The result prints an immutable `sha256:...` pin and a v1 artifact record. Exact bytes
are retained under:

```text
data/snapshots/v1/sha256/<snapshot-digest>/
├── dataset.parquet
├── metadata.json
└── record.json
```

The snapshot digest binds the exact Parquet and sidecar hashes, Arrow schema and Parquet
metadata, sidecar units, mutable kind/name identity, and ordered parent references. The
directory is content addressed: recapture is idempotent, while missing, partial, changed,
or conflicting content is rejected. Snapshot capture copies and then re-hashes both
sources under a shared lock; it does not query the table or contact a provider.

Large bytes stay under ignored `data/`. To transfer a snapshot, copy the **complete**
digest directory and verify it by reading through `Store.read_snapshot` or the CLI. Do
not commit Parquet snapshots to Git.

## Exact parents, export, and trace

Derived data must name each exact parent record, not merely repeat the mutable lineage
name:

```sh
uv run astrolabe --data-dir data snapshot capture widebin_pairs_baseline \
  --kind catalog \
  --parent data/snapshots/v1/sha256/<stars-pin>/record.json
```

Capture rejects unresolved lineage by default. `--allow-unresolved` exists only to retain
a limited historical snapshot; the resulting record says which parents are pending and
cannot be exported as a reproducible run input. A present-day parent hash never replaces
an absent historical parent or proves what an old run consumed.

```sh
# Inspect the exact local parent chain.
uv run astrolabe --data-dir data snapshot trace sha256:<digest>

# Export exact record/revision/source pins; output must not already exist.
uv run astrolabe --data-dir data snapshot manifest \
  --record data/snapshots/v1/sha256/<digest>/record.json \
  --output /tmp/run-input-manifest.json

# Validate against the one shared authority.
uv run orbit-research validate /tmp/run-input-manifest.json
```

Contract v1 requires clean Git provenance before a reference can be marked `resolved`.
Dataset bytes deliberately live outside Git, so owner-local snapshot references remain
`pending` even though their IDs, revisions, byte hashes, and locators are exact. This v1
generic reconciliation limitation is recorded in the migration; Astrolabe does not create
a second resolution schema to conceal it.

Read or restore a retained generation explicitly:

```sh
uv run astrolabe --data-dir data snapshot read sha256:<digest> --out /tmp/copy.parquet
uv run astrolabe --data-dir data snapshot restore sha256:<digest> --overwrite
```

Restore republishes the exact retained pair to its legacy mutable alias. It never removes
the immutable snapshot or its evidence.

## Baseline migration

`research/datasets/` is the versioned owner migration at Astrolabe revision
`90f5b58890da36c44286a4edbde7eead879410a8`:

- `inventory.json` is the read-only live inventory: 27 sidecars, 27 matching Parquet
  files, 27 mapped pairs, no pair exceptions, and 11 lineage containers. This exactly
  matches the framework foundation inventory; there is no live-count difference.
- `records/` contains one v1 artifact revision for every existing pair. It preserves
  every parsed sidecar field, exact sidecar bytes, legacy aliases, Parquet/sidecar hashes,
  lineage entry, and limitation. These migration records say `external`, not retained:
  current hashes do not claim that immutable bytes have already been copied.
- `dataset-manifest.json` pins all 27 records. All references are valid pending records;
  none fabricates historical consumption.
- `migration.json` accounts for all records and aliases. It explicitly retains the null,
  never-persisted Gaia parent of `derived/field_eq180_xmatch`, all other uncertain
  historical lineage, and the v1 non-Git reconciliation gap.
- `wide-binary-chain-manifest.json` pins the exact Astrolabe apparatus source and expected
  local dataset identities plus the published Principia protocol/program revisions. The
  four local wide-binary data products were absent from the inspected live inventory and
  remain `missing`. Orrery has not yet published canonical record IDs/revisions, so the
  two exact source-revision/path targets remain explicitly pending in `migration.json`.

Reproduce the dry run without modifying the source data:

```sh
uv run astrolabe --data-dir data snapshot inventory --output /tmp/astrolabe-inventory.json
# Rebuild the reviewed baseline from its frozen inventory (the current post-migration
# Git revision is intentionally not substituted for the historical source revision).
uv run python scripts/migrate_research_records.py build \
  --inventory research/datasets/inventory.json --output /tmp/astrolabe-records
uv run python scripts/migrate_research_records.py check
uv run orbit-research validate research/datasets/dataset-manifest.json
uv run orbit-research validate research/datasets/wide-binary-chain-manifest.json
```

`build` fails if the revision or reviewed 27/27/11 accounting changes. Review and record
that difference instead of silently refreshing a historical migration.

## Compatibility and rollback

Legacy callers continue to use `Store.read`, SQL views, and the original kind/name paths.
The migration records are the scientific provenance authority; the sidecars are preserved
compatibility input. To prove rollback equivalence, export all 27 original sidecars to a
new directory and compare their recorded hashes:

```sh
uv run python scripts/migrate_research_records.py rollback \
  --output /tmp/astrolabe-legacy-sidecars
```

The command refuses an existing or in-checkout output. Reverting the task commit removes
the code/record cutover while ignored snapshot directories and any later evidence remain
untouched. No rollback command deletes snapshots, fetches observations, reruns science, or
writes a sibling repository.
