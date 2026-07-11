"""Ephemeris-vs-baseline positional residuals (ORB-10076).

Compares a Horizons state-vector ephemeris against a model trajectory (e.g. the
pure-Newtonian n-body baseline from orrery `lab/sims/solar-system-nbody`) for a
single target. Both tables must follow the exchange convention in
`docs/baseline-interface.md`: same heliocentric ICRF frame, positions in AU,
epochs as JD TDB evaluated on the *same* grid — this function refuses to
interpolate, so a mismatched grid is an error, never a silent resample.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

POSITION_COLUMNS: tuple[str, ...] = ("x_au", "y_au", "z_au")
EPOCH_COLUMN = "epoch_jd_tdb"


def ephemeris_residuals(
    ephemeris: Table,
    baseline: Table,
    *,
    epoch_tol_d: float = 1e-6,
) -> Table:
    """Per-epoch positional residuals (baseline − ephemeris) for one target.

    Both tables need `epoch_jd_tdb` + `x_au/y_au/z_au`. Rows are matched by
    sorted epoch; any pairwise epoch difference beyond `epoch_tol_d` days (or a
    row-count mismatch) raises — the baseline must be evaluated on the
    ephemeris grid exactly.

    Returns a Table with `epoch_jd_tdb`, `dx_au`, `dy_au`, `dz_au`, `dr_au`,
    and summary stats in `meta` (`n_epochs`, `rms_dr_au`, `max_dr_au`,
    `max_dr_epoch_jd_tdb`).
    """
    for label, table in (("ephemeris", ephemeris), ("baseline", baseline)):
        missing = [
            c for c in (EPOCH_COLUMN, *POSITION_COLUMNS) if c not in table.colnames
        ]
        if missing:
            raise ValueError(f"{label} table is missing columns {missing}")

    if len(ephemeris) != len(baseline):
        raise ValueError(
            f"epoch grids differ: ephemeris has {len(ephemeris)} epochs, "
            f"baseline has {len(baseline)}"
        )
    if len(ephemeris) == 0:
        raise ValueError("ephemeris table is empty")

    eph = ephemeris[np.argsort(np.asarray(ephemeris[EPOCH_COLUMN], dtype=float))]
    base = baseline[np.argsort(np.asarray(baseline[EPOCH_COLUMN], dtype=float))]

    epochs = np.asarray(eph[EPOCH_COLUMN], dtype=float)
    depoch = np.abs(epochs - np.asarray(base[EPOCH_COLUMN], dtype=float))
    if depoch.max() > epoch_tol_d:
        raise ValueError(
            f"epoch grids differ by up to {depoch.max():g} d (> {epoch_tol_d:g}); "
            "evaluate the baseline on the ephemeris epoch grid — no interpolation here"
        )

    out = Table()
    out[EPOCH_COLUMN] = epochs
    for axis, col in zip(("dx_au", "dy_au", "dz_au"), POSITION_COLUMNS, strict=True):
        out[axis] = np.asarray(base[col], dtype=float) - np.asarray(
            eph[col], dtype=float
        )
    dr = np.sqrt(out["dx_au"] ** 2 + out["dy_au"] ** 2 + out["dz_au"] ** 2)
    out["dr_au"] = dr

    out.meta["n_epochs"] = len(out)
    out.meta["rms_dr_au"] = float(np.sqrt(np.mean(np.asarray(dr) ** 2)))
    out.meta["max_dr_au"] = float(dr.max())
    out.meta["max_dr_epoch_jd_tdb"] = float(epochs[int(np.argmax(np.asarray(dr)))])
    out.meta["convention"] = (
        "baseline minus ephemeris; heliocentric ICRF, AU, JD TDB "
        "(docs/baseline-interface.md)"
    )
    return out
