"""M2: CLI wiring — fetch/query/list against a temp catalog, source seam mocked."""

from __future__ import annotations

import json

from astrolabe import cli
from astrolabe.sources.gaia import GaiaSource


def test_fetch_then_query_then_list(tmp_path, gaia_table, monkeypatch, capsys):
    monkeypatch.setattr(GaiaSource, "_run_adql", lambda self, adql: gaia_table)

    rc = cli.main(
        [
            "--data-dir",
            str(tmp_path),
            "fetch",
            "gaia",
            "--name",
            "stars",
            "--ra",
            "10.0",
            "--dec",
            "41.0",
            "--radius",
            "0.1",
        ]
    )
    assert rc == 0
    assert "wrote 4 rows" in capsys.readouterr().out

    rc = cli.main(
        [
            "--data-dir",
            str(tmp_path),
            "query",
            "SELECT COUNT(*) AS n FROM stars",
        ]
    )
    assert rc == 0
    assert "n" in capsys.readouterr().out

    rc = cli.main(["--data-dir", str(tmp_path), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "catalog/stars" in out and "gaia" in out


def test_fetch_requires_params(tmp_path, capsys):
    rc = cli.main(
        [
            "--data-dir",
            str(tmp_path),
            "fetch",
            "gaia",
            "--name",
            "x",
        ]
    )
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
    rc = cli.main(
        [
            "--data-dir",
            str(tmp_path),
            "fetch",
            "gaia",
            "--name",
            "s",
            "--params",
            '{"adql": "SELECT source_id, ra, dec FROM gaiadr3.gaia_source"}',
        ]
    )
    assert rc == 0
    assert captured["adql"].startswith("SELECT source_id")


def test_snapshot_cli_capture_trace_inventory_and_manifest(tmp_path, gaia_table, capsys):
    store = cli.Store(tmp_path)
    store.write(gaia_table, name="stars", source="gaia")
    rc = cli.main(["--data-dir", str(tmp_path), "snapshot", "capture", "stars"])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["created"] is True

    rc = cli.main(["--data-dir", str(tmp_path), "snapshot", "trace", captured["snapshot_digest"]])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["dataset"] == "stars"

    inventory_path = tmp_path / "inventory.json"
    rc = cli.main(
        [
            "--data-dir",
            str(tmp_path),
            "snapshot",
            "inventory",
            "--output",
            str(inventory_path),
        ]
    )
    assert rc == 0
    assert json.loads(inventory_path.read_text())["counts"]["mapped"] == 1
    capsys.readouterr()

    manifest_path = tmp_path / "manifest.json"
    rc = cli.main(
        [
            "--data-dir",
            str(tmp_path),
            "snapshot",
            "manifest",
            "--record",
            captured["record"],
            "--output",
            str(manifest_path),
        ]
    )
    assert rc == 0
    assert json.loads(manifest_path.read_text())["kind"] == "manifest"
