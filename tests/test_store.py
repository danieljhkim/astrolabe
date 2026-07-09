"""M1: store round-trip (Table -> Parquet -> DuckDB) and metadata sidecar."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from astrolabe.store import Store


def test_write_read_roundtrip(tmp_path, gaia_table):
    store = Store(tmp_path)
    meta = store.write(gaia_table, name="stars", source="gaia", query={"radius": 0.1})

    assert meta.n_rows == len(gaia_table)
    assert meta.source == "gaia"
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


def test_list_datasets(tmp_path, gaia_table):
    store = Store(tmp_path)
    assert store.list_datasets() == []
    store.write(gaia_table, name="b", source="gaia")
    store.write(gaia_table, name="a", source="gaia")
    assert store.list_datasets() == ["a", "b"]


def test_query_over_catalog(tmp_path, gaia_table):
    store = Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    result = store.query('SELECT COUNT(*) AS n FROM stars WHERE dec < 0')
    assert isinstance(result, Table)
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
    assert (tmp_path / "processed" / "stars.parquet").exists()


def test_read_missing_raises(tmp_path):
    store = Store(tmp_path)
    try:
        store.read("nope")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
