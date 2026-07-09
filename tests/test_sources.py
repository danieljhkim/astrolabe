"""M2/M4: source adapters — protocol conformance + normalization, no network.

The network seam of each adapter is monkeypatched to return a fixture Table.
"""

from __future__ import annotations

import pytest

from astrolabe.sources import SOURCE_NAMES, get_source
from astrolabe.sources.base import STANDARD_COLUMNS, Source, ensure_standard_columns
from astrolabe.sources.gaia import GaiaSource
from astrolabe.sources.sdss import SDSSSource


def test_registry_resolves_all_names():
    for name in SOURCE_NAMES:
        src = get_source(name)
        assert isinstance(src, Source)  # runtime_checkable protocol
        assert src.name == name


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        get_source("hubble")


def test_gaia_cone_adql_built(gaia_table, monkeypatch):
    captured = {}

    def fake_run(self, adql):
        captured["adql"] = adql
        return gaia_table

    monkeypatch.setattr(GaiaSource, "_run_adql", fake_run)
    src = GaiaSource()
    out = src.query({"ra": 10.0, "dec": 41.0, "radius": 0.1, "limit": 50})

    assert "CONTAINS" in captured["adql"]
    assert "TOP 50" in captured["adql"]
    assert all(c in out.colnames for c in STANDARD_COLUMNS)


def test_gaia_adql_passthrough(gaia_table, monkeypatch):
    captured = {}

    def fake_run(self, adql):
        captured["adql"] = adql
        return gaia_table

    monkeypatch.setattr(GaiaSource, "_run_adql", fake_run)
    GaiaSource().query({"adql": "SELECT source_id, ra, dec FROM gaiadr3.gaia_source"})
    assert captured["adql"].startswith("SELECT source_id")


def test_gaia_cone_requires_coords(monkeypatch):
    monkeypatch.setattr(GaiaSource, "_run_adql", lambda self, adql: None)
    with pytest.raises(ValueError):
        GaiaSource().query({"ra": 10.0})  # missing dec/radius


def test_sdss_normalizes_objid(sdss_table, monkeypatch):
    monkeypatch.setattr(SDSSSource, "_run_cone", lambda self, params: sdss_table)
    out = SDSSSource().query({"ra": 10.0, "dec": 41.0, "radius": 0.1})
    assert "source_id" in out.colnames
    assert "objid" not in out.colnames
    assert all(c in out.colnames for c in STANDARD_COLUMNS)


def test_ensure_standard_columns_requires(gaia_table):
    ensure_standard_columns(gaia_table, require=True)  # ok
    bad = gaia_table.copy()
    bad.remove_column("dec")
    with pytest.raises(ValueError):
        ensure_standard_columns(bad, require=True)
