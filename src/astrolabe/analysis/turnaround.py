"""Turnaround / zero-velocity radius predictions vs measured R_ta (ORB-10754).

Pure functions over literature-compilation Tables. Cosmology is explicit in
query/meta so kepler can recompute at alternate H0 / Ω_m without re-fetching.

Predictions (all for a stated H0, G = CODATA via astropy):

- **shear-consumption / 2GM/H² scale** (task claim):
      R_sc = (2 G M / H0²)^{1/3}

- **Pavlidou–Tomaras ΛCDM absolute upper bound** on turnaround
  (Pavlidou & Tomaras 2014, JCAP 09, 020; arXiv:1310.1920):
      R_pt = (3 G M / (Λ c²))^{1/3} = (G M / (Ω_Λ H0²))^{1/3}
  with Λ c² = 3 Ω_Λ H0² in a flat universe.

- **Standard ΛCDM zero-velocity surface** (Lynden-Bell / Sandage with Λ):
      M = (π² / 8 G) R0³ H0² / f(Ω_m)²
  inverted for R0 given M. f(Ω_m) from Kashibadze & Karachentsev 2018
  (eq. 3; Planck Ω_m = 0.315 → M/Msun = 1.95e12 (R0/Mpc)³ at H0=67.3).

H0 sensitivity: primary at `h0_kms` (default 70); also report R_sc at 67 and 73.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.table import Table

# Default cosmology for primary prediction columns (stated in meta + sidecar).
DEFAULT_H0_KMS = 70.0
DEFAULT_OMEGA_M = 0.315
DEFAULT_OMEGA_LAMBDA = 0.685
H0_SENSITIVITY = (67.0, 73.0)

COLUMN_SEMANTICS: dict[str, str] = {
    "r_sc_mpc": "R_sc = (2GM/H0^2)^{1/3} using mass_analysis, Mpc",
    "r_sc_err_mpc": "propagated R_sc uncertainty from mass_analysis_err only, Mpc",
    "r_sc_indep_mpc": "R_sc using mass_indep when available, else nan, Mpc",
    "r_pt_max_mpc": "Pavlidou-Tomaras R_ta,max = (GM/(Omega_L H0^2))^{1/3}, Mpc",
    "r_lcdm_zv_mpc": "LCDM zero-velocity prediction inverted from f(Omega_m) M-R0, Mpc",
    "ratio_meas_sc": "r_ta / r_sc (mass_analysis)",
    "ratio_meas_sc_indep": "r_ta / r_sc_indep when mass_indep present",
    "ratio_meas_pt_max": "r_ta / r_pt_max (should be <= 1 under LCDM bound)",
    "ratio_meas_lcdm_zv": "r_ta / r_lcdm_zv (1 when M is circular from R0)",
    "r_sc_h67_mpc": "R_sc at H0=67 km/s/Mpc (mass_analysis), Mpc",
    "r_sc_h73_mpc": "R_sc at H0=73 km/s/Mpc (mass_analysis), Mpc",
    "ratio_meas_sc_h67": "r_ta / r_sc at H0=67",
    "ratio_meas_sc_h73": "r_ta / r_sc at H0=73",
    "h0_primary_kms": "H0 used for primary prediction columns, km/s/Mpc",
    "omega_m": "Omega_m assumed for LCDM zero-velocity inversion",
    "omega_lambda": "Omega_Lambda assumed for Pavlidou-Tomaras bound",
}


def f_omega_m(omega_m: float) -> float:
    """Dimensionless f(Ω_m) in the ΛCDM R0–M relation (Kashibadze & Karachentsev 2018).

    f runs from 1 (Ω_m→0) to 2/3 (Ω_m→1). Uses arcosh ≡ acosh.
    """
    if not 0.0 < omega_m < 1.0:
        raise ValueError(f"omega_m must be in (0,1), got {omega_m}")
    om = float(omega_m)
    return (1.0 - om) ** (-1) - 0.5 * om * (1.0 - om) ** (-1.5) * np.arccosh(
        2.0 / om - 1.0
    )


def r_sc_mpc(mass_msun: float | np.ndarray, h0_kms: float) -> np.ndarray:
    """R_sc = (2 G M / H0²)^{1/3} in Mpc."""
    m = np.asarray(mass_msun, dtype=float) * const.M_sun
    h0 = float(h0_kms) * u.km / u.s / u.Mpc
    r = (2.0 * const.G * m / h0**2) ** (1.0 / 3.0)
    return np.asarray(r.to(u.Mpc).value, dtype=float)


def r_pt_max_mpc(
    mass_msun: float | np.ndarray,
    h0_kms: float,
    omega_lambda: float = DEFAULT_OMEGA_LAMBDA,
) -> np.ndarray:
    """Pavlidou–Tomaras maximum turnaround radius in Mpc."""
    m = np.asarray(mass_msun, dtype=float) * const.M_sun
    h0 = float(h0_kms) * u.km / u.s / u.Mpc
    r = (const.G * m / (float(omega_lambda) * h0**2)) ** (1.0 / 3.0)
    return np.asarray(r.to(u.Mpc).value, dtype=float)


def r_lcdm_zv_mpc(
    mass_msun: float | np.ndarray,
    h0_kms: float,
    omega_m: float = DEFAULT_OMEGA_M,
) -> np.ndarray:
    """Invert M = (π²/8G) R0³ H0² / f(Ω_m)² for R0 in Mpc."""
    m = np.asarray(mass_msun, dtype=float) * const.M_sun
    h0 = float(h0_kms) * u.km / u.s / u.Mpc
    f = f_omega_m(omega_m)
    # M = (π²/8G) R³ H² / f²  →  R³ = M * 8 G * f² / (π² H²)
    r3 = m * 8.0 * const.G * f**2 / (np.pi**2 * h0**2)
    r = r3 ** (1.0 / 3.0)
    return np.asarray(r.to(u.Mpc).value, dtype=float)


def mass_from_r0_lcdm_msun(
    r0_mpc: float | np.ndarray,
    h0_kms: float,
    omega_m: float = DEFAULT_OMEGA_M,
) -> np.ndarray:
    """Forward M(R0) under the ΛCDM zero-velocity relation (for checks)."""
    r = np.asarray(r0_mpc, dtype=float) * u.Mpc
    h0 = float(h0_kms) * u.km / u.s / u.Mpc
    f = f_omega_m(omega_m)
    m = (np.pi**2 / (8.0 * const.G)) * r**3 * h0**2 / f**2
    return np.asarray(m.to(u.Msun).value, dtype=float)


def annotate_predictions(
    measured: Table,
    *,
    h0_kms: float = DEFAULT_H0_KMS,
    omega_m: float = DEFAULT_OMEGA_M,
    omega_lambda: float = DEFAULT_OMEGA_LAMBDA,
) -> Table:
    """Add prediction + ratio columns to a turnaround literature Table.

    Requires columns: r_ta_mpc, mass_analysis_1e12msun; optional
    mass_analysis_err_1e12msun, mass_indep_1e12msun.
    """
    required = ("r_ta_mpc", "mass_analysis_1e12msun")
    missing = [c for c in required if c not in measured.colnames]
    if missing:
        raise ValueError(f"measured table missing columns {missing}")

    out = measured.copy()
    m_an = np.asarray(out["mass_analysis_1e12msun"], dtype=float) * 1e12
    r_ta = np.asarray(out["r_ta_mpc"], dtype=float)

    if "mass_analysis_err_1e12msun" in out.colnames:
        m_err = np.asarray(out["mass_analysis_err_1e12msun"], dtype=float) * 1e12
    else:
        m_err = np.full_like(m_an, np.nan)

    r_sc = r_sc_mpc(m_an, h0_kms)
    # δR/R = (1/3) δM/M
    with np.errstate(divide="ignore", invalid="ignore"):
        r_sc_err = r_sc * (1.0 / 3.0) * (m_err / m_an)

    r_pt = r_pt_max_mpc(m_an, h0_kms, omega_lambda)
    r_zv = r_lcdm_zv_mpc(m_an, h0_kms, omega_m)

    out["r_sc_mpc"] = r_sc
    out["r_sc_err_mpc"] = r_sc_err
    out["r_pt_max_mpc"] = r_pt
    out["r_lcdm_zv_mpc"] = r_zv
    out["ratio_meas_sc"] = r_ta / r_sc
    out["ratio_meas_pt_max"] = r_ta / r_pt
    out["ratio_meas_lcdm_zv"] = r_ta / r_zv

    if "mass_indep_1e12msun" in out.colnames:
        m_ind = np.asarray(out["mass_indep_1e12msun"], dtype=float) * 1e12
        r_sc_i = r_sc_mpc(m_ind, h0_kms)
        # nan-safe: leave nan where mass_indep is nan
        out["r_sc_indep_mpc"] = r_sc_i
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ratio_meas_sc_indep"] = r_ta / r_sc_i
    else:
        out["r_sc_indep_mpc"] = np.full(len(out), np.nan)
        out["ratio_meas_sc_indep"] = np.full(len(out), np.nan)

    h67, h73 = H0_SENSITIVITY
    r67 = r_sc_mpc(m_an, h67)
    r73 = r_sc_mpc(m_an, h73)
    out["r_sc_h67_mpc"] = r67
    out["r_sc_h73_mpc"] = r73
    out["ratio_meas_sc_h67"] = r_ta / r67
    out["ratio_meas_sc_h73"] = r_ta / r73
    out["h0_primary_kms"] = np.full(len(out), float(h0_kms))
    out["omega_m"] = np.full(len(out), float(omega_m))
    out["omega_lambda"] = np.full(len(out), float(omega_lambda))

    out.meta["turnaround"] = {
        "h0_primary_kms": float(h0_kms),
        "h0_sensitivity_kms": list(H0_SENSITIVITY),
        "omega_m": float(omega_m),
        "omega_lambda": float(omega_lambda),
        "f_omega_m": float(f_omega_m(omega_m)),
        "r_sc_definition": "(2GM/H0^2)^(1/3)",
        "r_pt_max_definition": "(GM/(Omega_Lambda H0^2))^(1/3)  [Pavlidou & Tomaras 2014]",
        "r_lcdm_zv_definition": (
            "invert M=(pi^2/8G) R0^3 H0^2 / f(Omega_m)^2 "
            "(Kashibadze & Karachentsev 2018)"
        ),
        "note_h0_scaling": (
            "R_sc and R_pt scale as H0^{-2/3}; raising H0 from 67 to 73 "
            "shrinks R_pred by (67/73)^{2/3} ≈ 0.945 (~5.5% smaller)"
        ),
    }
    return out


def prediction_meta(
    *,
    h0_kms: float = DEFAULT_H0_KMS,
    omega_m: float = DEFAULT_OMEGA_M,
    omega_lambda: float = DEFAULT_OMEGA_LAMBDA,
) -> dict[str, Any]:
    """Sidecar-friendly cosmology block."""
    return {
        "h0_primary_kms": float(h0_kms),
        "h0_sensitivity_kms": list(H0_SENSITIVITY),
        "omega_m": float(omega_m),
        "omega_lambda": float(omega_lambda),
        "f_omega_m": float(f_omega_m(omega_m)),
    }
