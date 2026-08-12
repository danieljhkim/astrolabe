"""Deliver the turnaround-radii literature compilation (ORB-10754).

Offline: loads the curated `turnaround` Source table, annotates shear-consumption
and ΛCDM predictions, and writes:

    catalog/turnaround_radii   measured R0/R_ta + masses + citations + ratios

Sidecar `semantics` documents every column for the principia study-note consumer
(kepler). Exchange contract: docs/turnaround-interface.md.

Usage: uv run python scripts/deliver_turnaround.py [--data-dir data]
                                                   [--h0 70]
"""

from __future__ import annotations

import argparse

from astrolabe.analysis.turnaround import (
    COLUMN_SEMANTICS as PRED_SEMANTICS,
)
from astrolabe.analysis.turnaround import (
    DEFAULT_H0_KMS,
    DEFAULT_OMEGA_LAMBDA,
    DEFAULT_OMEGA_M,
    annotate_predictions,
    prediction_meta,
)
from astrolabe.sources import get_source
from astrolabe.sources.turnaround import COLUMN_SEMANTICS as LIT_SEMANTICS
from astrolabe.store import Store

DATASET_NAME = "turnaround_radii"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--h0",
        type=float,
        default=DEFAULT_H0_KMS,
        help=f"primary H0 in km/s/Mpc (default {DEFAULT_H0_KMS})",
    )
    parser.add_argument("--omega-m", type=float, default=DEFAULT_OMEGA_M)
    parser.add_argument("--omega-lambda", type=float, default=DEFAULT_OMEGA_LAMBDA)
    args = parser.parse_args()

    src = get_source("turnaround")
    store = Store(args.data_dir)

    measured = src.query({})
    table = annotate_predictions(
        measured,
        h0_kms=args.h0,
        omega_m=args.omega_m,
        omega_lambda=args.omega_lambda,
    )

    semantics = {**LIT_SEMANTICS, **PRED_SEMANTICS}
    query = {
        "source": "literature.turnaround",
        "task": "ORB-10754",
        **prediction_meta(
            h0_kms=args.h0,
            omega_m=args.omega_m,
            omega_lambda=args.omega_lambda,
        ),
        "n_systems": len(table),
        "n_circular_mass": int(sum(table["circularity_flag"])),
    }

    meta = store.write(
        table,
        name=DATASET_NAME,
        source=src.name,
        kind=src.kind,
        query=query,
        semantics=semantics,
    )
    print(f"wrote {meta.kind}/{meta.name}: {meta.n_rows} systems")
    print(
        f"  H0={args.h0} km/s/Mpc, Ωm={args.omega_m}, ΩΛ={args.omega_lambda}; "
        f"circular mass flags: {query['n_circular_mass']}/{meta.n_rows}"
    )

    # Compact summary for the operator / task record.
    print("\nsummary (r_ta / R_sc using mass_analysis | mass_indep):")
    for row in table:
        r_sc = row["ratio_meas_sc"]
        r_ind = row["ratio_meas_sc_indep"]
        r_zv = row["ratio_meas_lcdm_zv"]
        circ = "circ" if row["circularity_flag"] else "indep-mass"
        ind_s = f"{r_ind:.2f}" if r_ind == r_ind else "—"  # nan check
        print(
            f"  {row['source_id']:22s}  R0={row['r_ta_mpc']:.2f}±"
            f"{row['r_ta_err_mpc']:.2f} Mpc  "
            f"ratio_sc={r_sc:.2f}  ratio_sc_indep={ind_s}  "
            f"ratio_lcdm_zv={r_zv:.2f}  [{circ}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
