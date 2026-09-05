"""End-to-end wide-binary ṽ(g_N) study for ORB-10753.

Predeclared baseline policy is imported from analysis.wide_binaries (and the
task plan). This script:

1. Loads a local-volume Gaia DR3 star sample (network) or a synthetic field
   when TAP is unavailable / --synthetic is set.
2. Selects pairs under the locked cuts; estimates chance alignment.
3. Bins ṽ vs g_N against a thermal-eccentricity Newtonian Monte Carlo.
4. Builds the sensitivity table over ecc prior × RUWE cut.
5. Writes catalog + derived datasets to the Store with full provenance.

Usage:
    uv run python scripts/run_wide_binary_study.py [--data-dir data]
    uv run python scripts/run_wide_binary_study.py --synthetic --data-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from astropy.table import Table

from astrolabe.analysis.wide_binaries import (
    BASELINE_CUTS,
    BASELINE_ECC_PRIOR,
    COLUMN_SEMANTICS_BINNED,
    COLUMN_SEMANTICS_PAIRS,
    DEFAULT_G_EDGES,
    binned_vtilde,
    chance_alignment_rate,
    make_synthetic_star_field,
    newtonian_mock_pairs,
    select_wide_pairs,
    sensitivity_table,
)
from astrolabe.sources.wide_binaries import LOCAL_VOLUME_ADQL_TEMPLATE, WideBinarySource
from astrolabe.store import Store

STAR_DS = "widebin_gaia_dr3_d200"
PAIRS_DS = "widebin_pairs_baseline"
BINNED_DS = "widebin_vtilde_gn"
SENS_DS = "widebin_sensitivity"


def _fetch_stars(
    limit: int,
    *,
    ra0: float = 180.0,
    dec0: float = 40.0,
    radius_deg: float = 30.0,
) -> tuple[Table, dict[str, Any]]:
    src = WideBinarySource()
    params = {
        "mode": "local_volume",
        "limit": limit,
        "parallax_min_mas": BASELINE_CUTS["parallax_min_mas"],
        "ruwe_max": BASELINE_CUTS["ruwe_max"],
        "parallax_over_error_min": BASELINE_CUTS["parallax_over_error_min"],
        "g_mag_max": BASELINE_CUTS["g_mag_max"],
        "ra0": ra0,
        "dec0": dec0,
        "radius_deg": radius_deg,
    }
    table = src.query(params)
    # Force string source_ids for parquet-stable pair tables.
    if "source_id" in table.colnames:
        table["source_id"] = [str(x) for x in table["source_id"]]
    query = {
        "mode": "local_volume",
        "params": params,
        "adql": table.meta.get("wide_binaries_query", {}).get("adql"),
        "doi_gaia_dr3": "10.5270/esa-1ugzkg7",
        "sample_origin": "gaia_tap",
        "footprint": {"ra0": ra0, "dec0": dec0, "radius_deg": radius_deg},
    }
    return table, query


def _synthetic_stars(n_pairs: int, n_field: int, seed: int) -> tuple[Table, dict[str, Any]]:
    table = make_synthetic_star_field(
        n_pairs=n_pairs,
        n_field=n_field,
        seed=seed,
        ecc_prior=BASELINE_ECC_PRIOR,
    )
    adql = LOCAL_VOLUME_ADQL_TEMPLATE.format(
        top_clause="",
        columns="source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, "
                "pmdec, pmdec_error, phot_g_mean_mag, bp_rp, ruwe, "
                "radial_velocity, radial_velocity_error",
        table="gaiadr3.gaia_source",
        parallax_min_mas=BASELINE_CUTS["parallax_min_mas"],
        parallax_over_error_min=BASELINE_CUTS["parallax_over_error_min"],
        ruwe_max=BASELINE_CUTS["ruwe_max"],
        g_mag_max=BASELINE_CUTS["g_mag_max"],
        ra0=180.0,
        dec0=40.0,
        radius_deg=30.0,
    )
    query = {
        "mode": "synthetic_fallback",
        "adql_intended": adql,
        "doi_gaia_dr3": "10.5270/esa-1ugzkg7",
        "sample_origin": "synthetic_newtonian_star_field",
        "synthetic": table.meta.get("synthetic_star_field"),
        "warning": (
            "TAP fetch unavailable or --synthetic set. Results are a pipeline "
            "validation under a planted Newtonian field — not a Gaia DR3 science "
            "verdict. Re-run without --synthetic when TAP works for the real probe."
        ),
    }
    return table, query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip TAP; use planted Newtonian star field")
    parser.add_argument("--limit", type=int, default=0,
                        help="TOP N for Gaia ADQL (0 = full cone, no TOP)")
    parser.add_argument("--ra0", type=float, default=180.0)
    parser.add_argument("--dec0", type=float, default=40.0)
    parser.add_argument("--radius-deg", type=float, default=30.0)
    parser.add_argument("--max-stars", type=int, default=12000,
                        help="cap stars after quality cuts before spherical radius search")
    parser.add_argument("--n-mock", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", default="",
                        help="optional path to write a machine-readable summary")
    args = parser.parse_args(argv)

    # ---- 1. stars --------------------------------------------------------
    origin = "synthetic"
    if args.synthetic:
        stars, star_query = _synthetic_stars(400, 600, args.seed)
    else:
        try:
            print(
                f"fetching Gaia DR3 local-volume cone "
                f"(ra={args.ra0}, dec={args.dec0}, R={args.radius_deg}°) via TAP...",
                flush=True,
            )
            stars, star_query = _fetch_stars(
                args.limit,
                ra0=args.ra0,
                dec0=args.dec0,
                radius_deg=args.radius_deg,
            )
            origin = "gaia_tap"
            print(f"  got {len(stars)} stars", flush=True)
        except Exception as exc:  # network / TAP failures
            print(f"TAP fetch failed ({exc!r}); falling back to synthetic field",
                  flush=True)
            traceback.print_exc(file=sys.stderr)
            stars, star_query = _synthetic_stars(400, 600, args.seed)
            origin = "synthetic_fallback"

    store = Store(args.data_dir)
    meta_stars = store.write(
        stars,
        name=STAR_DS,
        source="wide_binaries" if origin == "gaia_tap" else "wide_binaries.synthetic",
        kind="catalog",
        query=star_query,
        semantics={
            "source_id": "Gaia DR3 source_id",
            "ra": "RA, deg",
            "dec": "Dec, deg",
            "parallax": "parallax, mas",
            "pmra": "pmra (incl. cos dec), mas/yr",
            "pmdec": "pmdec, mas/yr",
            "phot_g_mean_mag": "G magnitude",
            "ruwe": "renormalised unit weight error",
        },
    )
    print(f"wrote catalog/{STAR_DS}: {meta_stars.n_rows} rows ({origin})")

    # ---- 2. baseline pairs (LOCKED cuts) ---------------------------------
    pair_kwargs = {
        "ruwe_max": BASELINE_CUTS["ruwe_max"],
        "parallax_over_error_min": BASELINE_CUTS["parallax_over_error_min"],
        "g_mag_max": BASELINE_CUTS["g_mag_max"],
        "parallax_min_mas": BASELINE_CUTS["parallax_min_mas"],
        "theta_min_arcsec": BASELINE_CUTS["theta_min_arcsec"],
        "theta_max_arcsec": BASELINE_CUTS["theta_max_arcsec"],
        "s_min_kau": BASELINE_CUTS["s_min_kau"],
        "s_max_kau": BASELINE_CUTS["s_max_kau"],
        "parallax_sigma_max": BASELINE_CUTS["parallax_sigma_max"],
        "pm_escape_factor": BASELINE_CUTS["pm_escape_factor"],
        "rv_diff_max_kms": BASELINE_CUTS["rv_diff_max_kms"],
        "max_stars": args.max_stars,
    }
    pairs = select_wide_pairs(stars, **pair_kwargs)
    print(f"selected {len(pairs)} baseline pairs", flush=True)

    global_chance, chance_bins = chance_alignment_rate(
        stars, ra_shift_deg=0.5, g_edges=DEFAULT_G_EDGES, **pair_kwargs
    )
    print(f"chance-alignment global rate R = {global_chance:.3f}", flush=True)

    pair_query = {
        "baseline_cuts": BASELINE_CUTS,
        "pair_meta": pairs.meta.get("pair_cuts"),
        "chance_alignment": {
            "method": "shifted-field RA +0.5 deg",
            "global_rate": global_chance,
            "per_bin": [
                {
                    "g_lo": float(r["g_N_lo_ms2"]),
                    "g_hi": float(r["g_N_hi_ms2"]),
                    "n_real": int(r["n_real"]),
                    "n_shift": int(r["n_shift"]),
                    "r_chance": float(r["r_chance"]) if r["r_chance"] == r["r_chance"] else None,
                }
                for r in chance_bins
            ],
        },
        "triple_contamination_policy": {
            "ruwe_max": BASELINE_CUTS["ruwe_max"],
            "rv_diff_max_kms": BASELINE_CUTS["rv_diff_max_kms"],
            "f_triple_residual": BASELINE_CUTS["f_triple_residual"],
            "note": (
                "RUWE cut is the primary hierarchical-triple veto; residual "
                "triple fraction ~0.10 after cuts (El-Badry+2021-informed)"
            ),
        },
        "ecc_prior_baseline": BASELINE_ECC_PRIOR,
        "parent_stars": STAR_DS,
        "sample_origin": origin,
    }
    meta_pairs = store.write(
        pairs,
        name=PAIRS_DS,
        source="analysis.select_wide_pairs",
        kind="catalog",
        query=pair_query,
        lineage=[{"dataset": STAR_DS, "fetched_at": meta_stars.fetched_at}],
        semantics=COLUMN_SEMANTICS_PAIRS,
    )
    print(f"wrote catalog/{PAIRS_DS}: {meta_pairs.n_rows} pairs")

    # ---- 3. Newtonian mock + binned statistic ----------------------------
    mock = newtonian_mock_pairs(
        args.n_mock,
        ecc_prior=BASELINE_ECC_PRIOR,
        seed=args.seed,
        s_min_kau=BASELINE_CUTS["s_min_kau"],
        s_max_kau=BASELINE_CUTS["s_max_kau"],
    )
    binned = binned_vtilde(
        pairs, mock, g_edges=DEFAULT_G_EDGES, chance_per_bin=chance_bins, min_pairs=5
    )
    binned_query = {
        "statistic": "median vtilde = dv_sky / v_circ(s_proj, M_tot)",
        "g_edges_ms2": list(map(float, DEFAULT_G_EDGES)),
        "ecc_prior": BASELINE_ECC_PRIOR,
        "baseline_cuts": BASELINE_CUTS,
        "mock": mock.meta.get("newtonian_mock"),
        "binned_meta": binned.meta.get("binned_vtilde"),
        "sample_origin": origin,
    }
    meta_binned = store.write(
        binned,
        name=BINNED_DS,
        source="analysis.binned_vtilde",
        kind="derived",
        query=binned_query,
        lineage=[
            {"dataset": PAIRS_DS, "fetched_at": meta_pairs.fetched_at},
            {"dataset": STAR_DS, "fetched_at": meta_stars.fetched_at},
        ],
        semantics=COLUMN_SEMANTICS_BINNED,
    )
    print(f"wrote derived/{BINNED_DS}: {meta_binned.n_rows} bins")

    # ---- 4. sensitivity --------------------------------------------------
    pairs_ruwe12 = select_wide_pairs(stars, **{**pair_kwargs, "ruwe_max": 1.2})
    sens = sensitivity_table(
        pairs_cache={
            "1.4": pairs,
            "1.2": pairs_ruwe12,
        },
        n_mock=args.n_mock,
        seed=args.seed,
    )
    meta_sens = store.write(
        sens,
        name=SENS_DS,
        source="analysis.sensitivity_table",
        kind="derived",
        query=dict(sens.meta.get("sensitivity", {})),
        lineage=[
            {"dataset": PAIRS_DS, "fetched_at": meta_pairs.fetched_at},
            {"dataset": STAR_DS, "fetched_at": meta_stars.fetched_at},
        ],
        semantics={
            "ecc_prior": "eccentricity prior for Newtonian mock (thermal|flat)",
            "ruwe_max": "RUWE cut used for pair selection",
            "regime": "low_g (g_N < 1.2e-10) or high_g",
            "vtilde_med": "median scaled velocity in regime",
            "vtilde_mock_med": "Newtonian mock median in same regime",
        },
    )
    print(f"wrote derived/{SENS_DS}: {meta_sens.n_rows} rows")

    # ---- 5. printable summary --------------------------------------------
    summary = {
        "sample_origin": origin,
        "n_stars": int(len(stars)),
        "n_pairs_baseline": int(len(pairs)),
        "chance_alignment_global": global_chance,
        "f_triple_residual": BASELINE_CUTS["f_triple_residual"],
        "ecc_prior_baseline": BASELINE_ECC_PRIOR,
        "baseline_cuts": BASELINE_CUTS,
        "pair_selection": pairs.meta.get("pair_cuts"),
        "datasets": {
            "stars": f"catalog/{STAR_DS}",
            "pairs": f"catalog/{PAIRS_DS}",
            "binned": f"derived/{BINNED_DS}",
            "sensitivity": f"derived/{SENS_DS}",
        },
        "vtilde_gn_table": [
            {
                "g_N_mid_ms2": float(r["g_N_mid_ms2"]),
                "n_pairs": int(r["n_pairs"]),
                "vtilde_med": float(r["vtilde_med"]),
                "vtilde_err": float(r["vtilde_err"]),
                "vtilde_mock_med": float(r["vtilde_mock_med"]),
                "vtilde_mock_lo": float(r["vtilde_mock_lo"]),
                "vtilde_mock_hi": float(r["vtilde_mock_hi"]),
                "r_chance": float(r["r_chance"]) if r["r_chance"] == r["r_chance"] else None,
            }
            for r in binned
        ],
        "sensitivity_table": [
            {
                "ecc_prior": str(r["ecc_prior"]),
                "ruwe_max": float(r["ruwe_max"]),
                "regime": str(r["regime"]),
                "n_pairs": int(r["n_pairs"]),
                "vtilde_med": float(r["vtilde_med"]),
                "vtilde_err": float(r["vtilde_err"]),
                "vtilde_mock_med": float(r["vtilde_mock_med"]),
            }
            for r in sens
        ],
    }

    print("\n=== ṽ(g_N) baseline (thermal ecc, ruwe<1.4) ===")
    print(
        f"{'g_N_mid':>12} {'n':>5} {'ṽ_med':>8} {'err':>8} "
        f"{'mock':>8} {'p16':>8} {'p84':>8} {'R_ch':>8}"
    )
    for row in summary["vtilde_gn_table"]:
        print(
            f"{row['g_N_mid_ms2']:12.3e} {row['n_pairs']:5d} "
            f"{row['vtilde_med']:8.3f} {row['vtilde_err']:8.3f} "
            f"{row['vtilde_mock_med']:8.3f} {row['vtilde_mock_lo']:8.3f} "
            f"{row['vtilde_mock_hi']:8.3f} "
            f"{(row['r_chance'] if row['r_chance'] is not None else float('nan')):8.3f}"
        )

    print("\n=== sensitivity (contested choices) ===")
    print(f"{'ecc':>8} {'ruwe':>6} {'regime':>7} {'n':>5} {'ṽ':>8} {'mock':>8}")
    for row in summary["sensitivity_table"]:
        print(
            f"{row['ecc_prior']:>8} {row['ruwe_max']:6.1f} {row['regime']:>7} "
            f"{row['n_pairs']:5d} {row['vtilde_med']:8.3f} {row['vtilde_mock_med']:8.3f}"
        )

    out_path = args.summary_json or str(
        Path(args.data_dir) / "scratch" / "widebin_study_summary.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary JSON → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
