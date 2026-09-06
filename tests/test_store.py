"""M1: store round-trip (Table -> Parquet -> DuckDB) and metadata sidecar.

Also covers the kind-partitioned layout: datasets live under
data/processed/<kind>/ (kind in store.KINDS), derived datasets carry lineage,
and Store.query exposes each dataset as a schema-qualified DuckDB view.
"""

from __future__ import annotations

import json
import os

import duckdb
import numpy as np
import pytest
from astropy.table import Table

from astrolabe.store import Store


def test_write_read_roundtrip(tmp_path, gaia_table):
    store = Store(tmp_path)
    meta = store.write(gaia_table, name="stars", source="gaia", query={"radius": 0.1})

    assert meta.n_rows == len(gaia_table)
    assert meta.source == "gaia"
    assert meta.kind == "catalog"  # default kind
    assert "ra" in meta.columns

    back = store.read("stars")
    assert set(back.colnames) == set(gaia_table.colnames)
    assert len(back) == len(gaia_table)
    np.testing.assert_allclose(np.asarray(back["ra"]), np.asarray(gaia_table["ra"]))


def test_metadata_sidecar_persisted(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia", query={"radius": 0.1})
    meta = store.read_meta("stars")
    assert meta.name == "stars"
    assert meta.query == {"radius": 0.1}
    assert meta.fetched_at.endswith("+00:00")  # ISO-8601 UTC


def test_kind_partitioned_paths(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    store.write(gaia_table, name="mars_2026", source="horizons", kind="ephemeris")
    assert (tmp_path / "processed" / "catalog" / "stars.parquet").exists()
    assert (tmp_path / "processed" / "ephemeris" / "mars_2026.parquet").exists()
    assert (tmp_path / "processed" / "ephemeris" / "mars_2026.json").exists()


def test_write_unknown_kind_raises(tmp_path, gaia_table):
    store = Store(tmp_path)
    with pytest.raises(ValueError, match="unknown dataset kind"):
        store.write(gaia_table, name="x", source="gaia", kind="misc")


def test_derived_lineage_roundtrip(tmp_path, gaia_table):
    store = Store(tmp_path)
    lineage = [{"dataset": "stars", "fetched_at": "2026-07-09T00:00:00+00:00"}]
    store.write(
        gaia_table,
        name="stars_xmatch",
        source="analysis.crossmatch",
        kind="derived",
        lineage=lineage,
    )
    meta = store.read_meta("stars_xmatch")
    assert meta.kind == "derived"
    assert meta.lineage == lineage


def test_semantics_sidecar_roundtrip(tmp_path, gaia_table):
    store = Store(tmp_path)
    semantics = {"ra": "right ascension, deg", "dec": "declination, deg"}
    store.write(gaia_table, name="stars", source="gaia", semantics=semantics)
    assert store.read_meta("stars").semantics == semantics
    # Sidecars written before the field existed still load (default None).
    store.write(gaia_table, name="stars_old", source="gaia")
    meta_path = tmp_path / "processed" / "catalog" / "stars_old.json"
    raw = json.loads(meta_path.read_text())
    raw.pop("semantics")
    meta_path.write_text(json.dumps(raw))
    assert store.read_meta("stars_old").semantics is None


def test_same_name_across_kinds_needs_kind(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="mars", source="gaia", kind="catalog")
    store.write(gaia_table, name="mars", source="horizons", kind="ephemeris")
    with pytest.raises(ValueError, match="multiple kinds"):
        store.read("mars")
    back = store.read("mars", kind="ephemeris")
    assert len(back) == len(gaia_table)


def test_list_datasets(tmp_path, gaia_table):
    store = Store(tmp_path)
    assert store.list_datasets() == []
    store.write(gaia_table, name="b", source="gaia")
    store.write(gaia_table, name="a", source="gaia")
    store.write(gaia_table, name="c", source="horizons", kind="ephemeris")
    assert store.list_datasets() == ["a", "b", "c"]
    assert store.list_datasets(kind="catalog") == ["a", "b"]
    assert store.datasets() == [
        ("catalog", "a"),
        ("catalog", "b"),
        ("ephemeris", "c"),
    ]


def test_query_over_catalog(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    result = store.query("SELECT COUNT(*) AS n FROM stars WHERE dec < 0")
    assert isinstance(result, Table)
    assert int(result["n"][0]) == 2


def test_query_schema_qualified_views(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    store.write(gaia_table, name="mars_2026", source="horizons", kind="ephemeris")
    result = store.query(
        "SELECT (SELECT COUNT(*) FROM catalog.stars) AS n_cat, "
        "(SELECT COUNT(*) FROM ephemeris.mars_2026) AS n_eph"
    )
    assert int(result["n_cat"][0]) == len(gaia_table)
    assert int(result["n_eph"][0]) == len(gaia_table)


def test_query_ambiguous_name_only_schema_qualified(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="mars", source="gaia", kind="catalog")
    store.write(gaia_table[:2], name="mars", source="horizons", kind="ephemeris")
    # No unqualified alias when the name exists under more than one kind.
    with pytest.raises(duckdb.CatalogException):
        store.query("SELECT COUNT(*) FROM mars")
    result = store.query("SELECT COUNT(*) AS n FROM ephemeris.mars")
    assert int(result["n"][0]) == 2


def test_query_select_columns(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    result = store.query("SELECT source_id, ra FROM stars ORDER BY source_id LIMIT 1")
    assert set(result.colnames) == {"source_id", "ra"}
    assert int(result["source_id"][0]) == 1001


def test_raw_stored_separately(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia", raw=gaia_table)
    assert (tmp_path / "raw" / "gaia" / "stars.parquet").exists()
    assert (tmp_path / "processed" / "catalog" / "stars.parquet").exists()


def test_read_missing_raises(tmp_path):
    store = Store(tmp_path)
    try:
        store.read("nope")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_write_partial_publish_rolls_back_exact_old_pair(tmp_path, gaia_table, monkeypatch):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    data_path = store._processed_path("catalog", "stars")
    meta_path = store._meta_path("catalog", "stars")
    before = (data_path.read_bytes(), meta_path.read_bytes())
    real_replace = os.replace
    calls = 0

    def fail_second_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected metadata publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="metadata publish"):
        store.write(gaia_table[:1], name="stars", source="gaia")
    assert (data_path.read_bytes(), meta_path.read_bytes()) == before
