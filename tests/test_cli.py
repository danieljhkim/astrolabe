"""M2: CLI wiring — fetch/query/list against a temp catalog, source seam mocked."""

from __future__ import annotations

from astrolabe import cli
from astrolabe.sources.gaia import GaiaSource


def test_fetch_then_query_then_list(tmp_path, gaia_table, monkeypatch, capsys):
    monkeypatch.setattr(GaiaSource, "_run_adql", lambda self, adql: gaia_table)

    rc = cli.main([
        "--data-dir", str(tmp_path),
        "fetch", "gaia", "--name", "stars",
        "--ra", "10.0", "--dec", "41.0", "--radius", "0.1",
    ])
    assert rc == 0
    assert "wrote 4 rows" in capsys.readouterr().out

    rc = cli.main([
        "--data-dir", str(tmp_path),
        "query", "SELECT COUNT(*) AS n FROM stars",
    ])
    assert rc == 0
    assert "n" in capsys.readouterr().out

    rc = cli.main(["--data-dir", str(tmp_path), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "catalog/stars" in out and "gaia" in out


def test_fetch_requires_params(tmp_path, capsys):
    rc = cli.main([
        "--data-dir", str(tmp_path), "fetch", "gaia", "--name", "x",
    ])
    assert rc == 2
    assert "provide query params" in capsys.readouterr().err


def test_list_empty(tmp_path, capsys):
    rc = cli.main(["--data-dir", str(tmp_path), "list"])
    assert rc == 0
    assert "(no datasets)" in capsys.readouterr().out


def test_params_json_overrides_flags(tmp_path, gaia_table, monkeypatch, capsys):
    captured = {}

    def fake_run(self, adql):
        captured["adql"] = adql
        return gaia_table

    monkeypatch.setattr(GaiaSource, "_run_adql", fake_run)
    rc = cli.main([
        "--data-dir", str(tmp_path),
        "fetch", "gaia", "--name", "s",
        "--params", '{"adql": "SELECT source_id, ra, dec FROM gaiadr3.gaia_source"}',
    ])
    assert rc == 0
    assert captured["adql"].startswith("SELECT source_id")
