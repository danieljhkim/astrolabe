"""Wide-binary velocity statistics across the low-acceleration boundary.

Pure functions (Table in / Table out; no network, no I/O). Selection follows an
El-Badry & Rix-style local-volume gate; the scaled relative-velocity statistic

    ṽ = Δv_sky / v_circ(s, M_tot),   v_circ = √(G M_tot / s)

is binned against the internal Newtonian acceleration

    g_N = G (M1 + M2) / s²

with s the *projected* separation (standard practice; never deprojected here).

The eccentricity prior, quality cuts, and triple-contamination policy are
caller-supplied and recorded in output metadata — predeclare them before
looking at the binned result (ORB-10753). Comparison is always against a
forward-modeled Newtonian Monte Carlo with the same selection function, not an
analytic circular-orbit expectation.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

# -- physical constants (SI-consistent conversions) -----------------------
# G M_sun / AU^2 in m/s^2: acceleration of 1 Msun at 1 AU.
_G_MSUN_AU2_MS2 = 5.931e-3
# Circular speed of 1 Msun at 1 AU, km/s (Earth ≈ 29.78).
_V_CIRC_1MSUN_1AU_KMS = 29.7847
# Tangential speed (km/s) = _K * mu(mas/yr) * d(pc).
_K_MU_D = 4.74047e-3
# AU per parsec (definition: 1 AU subtends 1" at 1 pc).
_AU_PER_PC = 206264.806

EccPrior = Literal["thermal", "flat"]

# Baseline predeclared policy (mirrors the ORB-10753 plan). Contested
# alternatives are exercised only via explicit kwargs / sensitivity_table.
BASELINE_CUTS: dict[str, Any] = {
    "ruwe_max": 1.4,
    "parallax_over_error_min": 10.0,
    "g_mag_max": 18.0,
    "parallax_min_mas": 5.0,  # d < 200 pc
    "theta_min_arcsec": 1.5,
    "theta_max_arcsec": 3600.0,  # 1 deg
    "s_min_kau": 0.5,
    "s_max_kau": 50.0,
    "parallax_sigma_max": 3.0,
    "pm_escape_factor": 3.0,
    "rv_diff_max_kms": 20.0,
    "f_triple_residual": 0.10,
}
BASELINE_ECC_PRIOR: EccPrior = "thermal"

# Log-spaced g_N edges covering 1e-8 → 1e-11 m/s^2 (and a half-decade pad).
DEFAULT_G_EDGES = np.logspace(-11.5, -7.5, 9)

COLUMN_SEMANTICS_PAIRS: dict[str, str] = {
    "source_id_1": "Gaia source_id of primary (brighter G)",
    "source_id_2": "Gaia source_id of secondary",
    "ra_1": "primary RA, deg",
    "dec_1": "primary Dec, deg",
    "ra_2": "secondary RA, deg",
    "dec_2": "secondary Dec, deg",
    "parallax_mas": "mean pair parallax, mas",
    "distance_pc": "distance from mean parallax, pc",
    "theta_arcsec": "angular separation, arcsec",
    "s_kau": "projected physical separation, kau (1000 AU)",
    "s_au": "projected physical separation, AU",
    "m1_msun": "primary photometric mass, Msun (Pecaut & Mamajek 2013 M_G)",
    "m2_msun": "secondary photometric mass, Msun",
    "m_tot_msun": "M1+M2, Msun",
    "dv_kms": "sky-plane relative velocity from proper motions, km/s",
    "v_circ_kms": "circular speed sqrt(G M_tot / s) at projected s, km/s",
    "vtilde": "scaled relative velocity dv / v_circ (dimensionless)",
    "g_N_ms2": "internal Newtonian acceleration G M_tot / s^2, m/s^2",
    "ruwe_1": "primary RUWE",
    "ruwe_2": "secondary RUWE",
}

COLUMN_SEMANTICS_BINNED: dict[str, str] = {
    "g_N_lo_ms2": "bin lower edge in g_N, m/s^2",
    "g_N_hi_ms2": "bin upper edge in g_N, m/s^2",
    "g_N_mid_ms2": "log-mid bin center, m/s^2",
    "n_pairs": "number of pairs in bin",
    "vtilde_med": "median scaled velocity ṽ",
    "vtilde_err": "robust error on median 1.4826*MAD/sqrt(n)",
    "vtilde_mock_med": "median ṽ from Newtonian Monte Carlo (same selection)",
    "vtilde_mock_lo": "16th percentile of mock ṽ",
    "vtilde_mock_hi": "84th percentile of mock ṽ",
    "r_chance": "shifted-field chance-alignment rate N_shift/N_real in bin",
    "flag_high_chance": "1 if r_chance > 0.1",
}


# ---------------------------------------------------------------------------
# Photometric masses (Pecaut & Mamajek 2013 main-sequence M_G → M/Msun)
# ---------------------------------------------------------------------------
# Piecewise linear in log M vs M_G, calibrated to the PM13 table for dwarfs.
# Band: Gaia G (approx. via the published M_G sequence). Adequate for order-
# unity masses in a wide-binary acceleration test; not a spectroscopic mass.
_MG_KNOTS = np.array(
    [-2.0, 0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dtype=float
)
_LOGM_KNOTS = np.array(
    [1.0, 0.6, 0.3, 0.05, -0.05, -0.2, -0.5, -0.85, -1.15, -1.4, -1.6], dtype=float
)


def mass_from_abs_g(abs_g: np.ndarray) -> np.ndarray:
    """Main-sequence mass (Msun) from absolute G magnitude.

    Pecaut & Mamajek 2013-inspired piecewise log-linear map. Out-of-range M_G
    is clipped to the knot ends (very bright → ~10 Msun; very faint → ~0.025).
    """
    abs_g = np.asarray(abs_g, dtype=float)
    logm = np.interp(abs_g, _MG_KNOTS, _LOGM_KNOTS)
    return 10.0 ** logm


def absolute_g(g_mag: np.ndarray, parallax_mas: np.ndarray) -> np.ndarray:
    """Absolute G from apparent G and parallax (mas). Non-positive ϖ → NaN."""
    g_mag = np.asarray(g_mag, dtype=float)
    parallax_mas = np.asarray(parallax_mas, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = g_mag + 5.0 * np.log10(parallax_mas) - 10.0
    out = np.asarray(out, dtype=float)
    out[parallax_mas <= 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------
def _as_float(table: Table, col: str) -> np.ndarray:
    return np.asarray(table[col], dtype=float)


def _angular_sep_arcsec(
    ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray
) -> np.ndarray:
    """Great-circle separation in arcsec (vectorized; small-angle safe)."""
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)
    dra = ra2 - ra1
    # Vincenty-like stable formula
    num = (
        np.cos(dec2) * np.sin(dra)
    ) ** 2 + (
        np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra)
    ) ** 2
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    ang = np.arctan2(np.sqrt(num), den)  # radians
    return np.rad2deg(ang) * 3600.0


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return Cartesian unit vectors for finite ICRS-like coordinates."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float) % 360.0)
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


def _radius_candidates(
    primary_ra: np.ndarray,
    primary_dec: np.ndarray,
    secondary_ra: np.ndarray,
    secondary_dec: np.ndarray,
    theta_max_arcsec: float,
) -> tuple[list[tuple[int, int]], float, int, int]:
    """Find complete spherical-radius candidates without an NxN distance array.

    A chord radius on unit vectors is exactly equivalent to a great-circle
    radius.  The final Vincenty separation check remains authoritative.
    """
    primary_ok = np.isfinite(primary_ra) & np.isfinite(primary_dec)
    secondary_ok = np.isfinite(secondary_ra) & np.isfinite(secondary_dec)
    primary_idx = np.flatnonzero(primary_ok)
    secondary_idx = np.flatnonzero(secondary_ok)
    if not len(primary_idx) or not len(secondary_idx):
        return [], 0.0, int(len(primary_idx)), int(len(secondary_idx))

    theta_rad = np.deg2rad(theta_max_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(theta_rad / 2.0)
    start = perf_counter()
    tree = cKDTree(_unit_vectors(secondary_ra[secondary_idx], secondary_dec[secondary_idx]))
    neighbors = tree.query_ball_point(
        _unit_vectors(primary_ra[primary_idx], primary_dec[primary_idx]), chord_radius
    )
    elapsed = perf_counter() - start
    candidates = [
        (int(i), int(secondary_idx[j]))
        for i, matches in zip(primary_idx, neighbors, strict=True)
        for j in matches
    ]
    # SciPy does not promise a query_ball_point ordering across implementations.
    candidates.sort()
    return candidates, elapsed, int(len(primary_idx)), int(len(secondary_idx))


def _v_circ_kms(m_tot: np.ndarray, s_au: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _V_CIRC_1MSUN_1AU_KMS * np.sqrt(m_tot / s_au)


def _g_n_ms2(m_tot: np.ndarray, s_au: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _G_MSUN_AU2_MS2 * m_tot / s_au**2


def select_star_quality(
    stars: Table,
    *,
    ruwe_max: float = BASELINE_CUTS["ruwe_max"],
    parallax_over_error_min: float = BASELINE_CUTS["parallax_over_error_min"],
    g_mag_max: float = BASELINE_CUTS["g_mag_max"],
    parallax_min_mas: float = BASELINE_CUTS["parallax_min_mas"],
) -> Table:
    """Apply per-star quality + local-volume cuts; return a copy."""
    required = (
        "source_id", "ra", "dec", "parallax", "parallax_error",
        "pmra", "pmdec", "phot_g_mean_mag", "ruwe",
    )
    missing = [c for c in required if c not in stars.colnames]
    if missing:
        raise ValueError(f"star table missing columns {missing}")

    plx = _as_float(stars, "parallax")
    eplx = _as_float(stars, "parallax_error")
    with np.errstate(invalid="ignore", divide="ignore"):
        poe = plx / eplx
    g = _as_float(stars, "phot_g_mean_mag")
    ruwe = _as_float(stars, "ruwe")
    keep = (
        np.isfinite(plx) & np.isfinite(eplx) & (eplx > 0)
        & (plx >= parallax_min_mas)
        & (poe >= parallax_over_error_min)
        & (ruwe < ruwe_max)
        & np.isfinite(g) & (g < g_mag_max)
        & np.isfinite(_as_float(stars, "pmra"))
        & np.isfinite(_as_float(stars, "pmdec"))
    )
    if "bp_rp" in stars.colnames:
        keep &= np.isfinite(_as_float(stars, "bp_rp"))
    out = stars[keep].copy()
    out.meta = dict(stars.meta) if stars.meta else {}
    out.meta["star_quality_cuts"] = {
        "ruwe_max": ruwe_max,
        "parallax_over_error_min": parallax_over_error_min,
        "g_mag_max": g_mag_max,
        "parallax_min_mas": parallax_min_mas,
        "n_in": int(len(stars)),
        "n_out": int(len(out)),
    }
    return out


def select_wide_pairs(
    stars: Table,
    *,
    ruwe_max: float = BASELINE_CUTS["ruwe_max"],
    parallax_over_error_min: float = BASELINE_CUTS["parallax_over_error_min"],
    g_mag_max: float = BASELINE_CUTS["g_mag_max"],
    parallax_min_mas: float = BASELINE_CUTS["parallax_min_mas"],
    theta_min_arcsec: float = BASELINE_CUTS["theta_min_arcsec"],
    theta_max_arcsec: float = BASELINE_CUTS["theta_max_arcsec"],
    s_min_kau: float = BASELINE_CUTS["s_min_kau"],
    s_max_kau: float = BASELINE_CUTS["s_max_kau"],
    parallax_sigma_max: float = BASELINE_CUTS["parallax_sigma_max"],
    pm_escape_factor: float = BASELINE_CUTS["pm_escape_factor"],
    rv_diff_max_kms: float = BASELINE_CUTS["rv_diff_max_kms"],
    ra_shift_deg: float = 0.0,
    max_stars: int | None = 8000,
) -> Table:
    """El-Badry-style pair selection from a Gaia-shaped star table.

    Candidate generation uses a :class:`scipy.spatial.cKDTree` over Cartesian
    unit vectors, so every pair within ``theta_max_arcsec`` on the sphere is
    considered, including across RA=0 and near the poles.  It does not form an
    NxN distance matrix.

    With ``ra_shift_deg == 0``, the catalog is matched to itself, self-matches
    are excluded, and each unordered source-id pair is considered once.  With
    a nonzero shift, the unshifted catalog is the primary field and a distinct
    RA-shifted copy is the secondary field.  Same-source matches are excluded;
    if both directed shifted matches for an unordered source-id pair pass all
    cuts, the lexicographically first candidate is retained.  This makes the
    chance-alignment control independent of candidate traversal order while
    retaining a shifted match whenever either direction qualifies.

    ``max_stars`` caps the post-quality sample (brightest first) before the
    radius search; set it to ``None`` only when the caller has accepted an
    uncapped sample.  Output coordinates remain the source coordinates; the
    shift used for the secondary search field is recorded in metadata.
    """
    clean = select_star_quality(
        stars,
        ruwe_max=ruwe_max,
        parallax_over_error_min=parallax_over_error_min,
        g_mag_max=g_mag_max,
        parallax_min_mas=parallax_min_mas,
    )
    n_input = len(stars)
    n_quality = len(clean)
    cap_applied = max_stars is not None and n_quality > max_stars
    if cap_applied:
        order = np.argsort(_as_float(clean, "phot_g_mean_mag"))
        clean = clean[order[:max_stars]]

    n = len(clean)
    if n < 2:
        empty = _empty_pairs_table()
        empty.meta["pair_cuts"] = {
            "n_input_stars": n_input,
            "n_quality_stars": n_quality,
            "n_capped_stars": n,
            "max_stars": max_stars,
            "cap_applied": cap_applied,
            "ra_shift_deg": ra_shift_deg,
            "candidate_generation": "cKDTree unit-vector spherical radius search",
            "n_radius_candidates": 0,
            "candidate_search_seconds": 0.0,
            "n_pairs": 0,
        }
        return empty

    ra = _as_float(clean, "ra")
    dec = _as_float(clean, "dec")
    plx = _as_float(clean, "parallax")
    eplx = _as_float(clean, "parallax_error")
    pmra = _as_float(clean, "pmra")
    pmdec = _as_float(clean, "pmdec")
    gmag = _as_float(clean, "phot_g_mean_mag")
    ruwe = _as_float(clean, "ruwe")
    sid = np.array([str(x) for x in clean["source_id"]], dtype=object)
    e_pmra = (
        _as_float(clean, "pmra_error")
        if "pmra_error" in clean.colnames
        else np.full(n, 0.05)
    )
    e_pmdec = (
        _as_float(clean, "pmdec_error")
        if "pmdec_error" in clean.colnames
        else np.full(n, 0.05)
    )
    has_rv = "radial_velocity" in clean.colnames
    rv = _as_float(clean, "radial_velocity") if has_rv else np.full(n, np.nan)

    # The shifted copy is a distinct secondary catalog.  Modulo is important
    # both for RA-wrap controls and for valid cKDTree unit vectors.
    ra2_all = (ra + ra_shift_deg) % 360.0

    rows: list[tuple[Any, ...]] = []
    candidates, search_seconds, n_primary_searchable, n_secondary_searchable = (
        _radius_candidates(ra, dec, ra2_all, dec, theta_max_arcsec)
    )
    is_shifted = ra_shift_deg != 0.0
    seen_source_pairs: set[tuple[str, str]] = set()

    for i, j in candidates:
        # A real catalog is an unordered self-match.  A shifted catalog is a
        # cross-match, but source-id self matches remain unphysical controls.
        if sid[i] == sid[j] or (not is_shifted and j <= i):
            continue
        # Angular separation uses shifted secondary coordinates.
        th = float(
            _angular_sep_arcsec(
                np.array([ra[i]]),
                np.array([dec[i]]),
                np.array([ra2_all[j]]),
                np.array([dec[j]]),
            )[0]
        )
        if th < theta_min_arcsec or th > theta_max_arcsec:
            continue
        # parallax consistency
        dplx = abs(plx[i] - plx[j])
        sig = np.sqrt(eplx[i] ** 2 + eplx[j] ** 2)
        if sig <= 0 or dplx > parallax_sigma_max * sig:
            continue
        plx_mean = 0.5 * (plx[i] + plx[j])
        if plx_mean <= 0:
            continue
        d_pc = 1000.0 / plx_mean
        s_au = th * d_pc  # arcsec * pc = AU
        s_kau = s_au / 1000.0
        if s_kau < s_min_kau or s_kau > s_max_kau:
            continue

        # photometric masses for escape gate
        mg_i = absolute_g(np.array([gmag[i]]), np.array([plx[i]]))[0]
        mg_j = absolute_g(np.array([gmag[j]]), np.array([plx[j]]))[0]
        if not (np.isfinite(mg_i) and np.isfinite(mg_j)):
            continue
        m_i = float(mass_from_abs_g(np.array([mg_i]))[0])
        m_j = float(mass_from_abs_g(np.array([mg_j]))[0])
        m_tot = m_i + m_j
        v_c = float(_v_circ_kms(np.array([m_tot]), np.array([s_au]))[0])
        # max circular-equivalent PM (mas/yr) for escape factor * v_circ
        # v = K * mu * d  =>  mu = v / (K * d)
        mu_max = (pm_escape_factor * v_c) / (_K_MU_D * d_pc)
        dpmra = pmra[i] - pmra[j]
        dpmdec = pmdec[i] - pmdec[j]
        dpm = np.hypot(dpmra, dpmdec)
        sig_pm = np.sqrt(e_pmra[i] ** 2 + e_pmra[j] ** 2 + e_pmdec[i] ** 2 + e_pmdec[j] ** 2)
        if dpm > mu_max + 3.0 * sig_pm:
            continue

        # RV consistency when both measured
        if has_rv and np.isfinite(rv[i]) and np.isfinite(rv[j]):
            if abs(rv[i] - rv[j]) > rv_diff_max_kms:
                continue

        source_pair = tuple(sorted((sid[i], sid[j])))
        if source_pair in seen_source_pairs:
            continue
        seen_source_pairs.add(source_pair)

        # Primary = brighter (lower G); source id gives a stable tie-break.
        if (gmag[i], sid[i]) <= (gmag[j], sid[j]):
            p, s, m_p, m_s = i, j, m_i, m_j
        else:
            p, s, m_p, m_s = j, i, m_j, m_i

        dv = _K_MU_D * dpm * d_pc
        v_circ = float(_v_circ_kms(np.array([m_p + m_s]), np.array([s_au]))[0])
        g_n = float(_g_n_ms2(np.array([m_p + m_s]), np.array([s_au]))[0])
        vtilde = dv / v_circ if v_circ > 0 else np.nan

        rows.append(
            (
                sid[p], sid[s],
                ra[p], dec[p], ra[s], dec[s],
                float(plx_mean), float(d_pc), float(th),
                float(s_kau), float(s_au),
                float(m_p), float(m_s), float(m_p + m_s),
                float(dv), float(v_circ), float(vtilde), float(g_n),
                float(ruwe[p]), float(ruwe[s]),
            )
        )

    rows.sort(key=lambda row: (str(row[0]), str(row[1]), row[8]))
    out = _pairs_table_from_rows(rows)
    out.meta["pair_cuts"] = {
        "ruwe_max": ruwe_max,
        "parallax_over_error_min": parallax_over_error_min,
        "g_mag_max": g_mag_max,
        "parallax_min_mas": parallax_min_mas,
        "theta_min_arcsec": theta_min_arcsec,
        "theta_max_arcsec": theta_max_arcsec,
        "s_min_kau": s_min_kau,
        "s_max_kau": s_max_kau,
        "parallax_sigma_max": parallax_sigma_max,
        "pm_escape_factor": pm_escape_factor,
        "rv_diff_max_kms": rv_diff_max_kms,
        "ra_shift_deg": ra_shift_deg,
        "max_stars": max_stars,
        "n_input_stars": n_input,
        "n_quality_stars": n_quality,
        "n_capped_stars": n,
        "cap_applied": cap_applied,
        "n_stars": n,
        "candidate_generation": "cKDTree unit-vector spherical radius search",
        "candidate_search_radius_arcsec": theta_max_arcsec,
        "n_primary_searchable": n_primary_searchable,
        "n_secondary_searchable": n_secondary_searchable,
        "n_radius_candidates": len(candidates),
        "candidate_search_seconds": search_seconds,
        "shifted_catalog_semantics": (
            "unshifted primary catalog vs RA-shifted secondary catalog; same-source "
            "matches excluded; unordered source pairs deduplicated after all cuts"
        ),
        "n_pairs": len(out),
        "mass_relation": "Pecaut & Mamajek 2013-inspired M_G (Gaia G) piecewise log-linear",
        "f_triple_residual": BASELINE_CUTS["f_triple_residual"],
    }
    return out


_PAIR_COLNAMES = (
    "source_id_1", "source_id_2",
    "ra_1", "dec_1", "ra_2", "dec_2",
    "parallax_mas", "distance_pc", "theta_arcsec",
    "s_kau", "s_au",
    "m1_msun", "m2_msun", "m_tot_msun",
    "dv_kms", "v_circ_kms", "vtilde", "g_N_ms2",
    "ruwe_1", "ruwe_2",
)


def _pairs_table_from_rows(rows: list[tuple[Any, ...]]) -> Table:
    """Build a parquet-friendly pairs table (string ids, plain float64)."""
    if not rows:
        return _empty_pairs_table()
    cols: dict[str, list[Any]] = {name: [] for name in _PAIR_COLNAMES}
    for row in rows:
        for name, val in zip(_PAIR_COLNAMES, row, strict=True):
            cols[name].append(val)
    data: dict[str, Any] = {
        "source_id_1": np.array([str(x) for x in cols["source_id_1"]], dtype="U32"),
        "source_id_2": np.array([str(x) for x in cols["source_id_2"]], dtype="U32"),
    }
    for name in _PAIR_COLNAMES[2:]:
        data[name] = np.asarray(cols[name], dtype=float)
    return Table(data)


def _empty_pairs_table() -> Table:
    data: dict[str, Any] = {
        "source_id_1": np.array([], dtype="U32"),
        "source_id_2": np.array([], dtype="U32"),
    }
    for name in _PAIR_COLNAMES[2:]:
        data[name] = np.array([], dtype=float)
    return Table(data)


def chance_alignment_rate(
    stars: Table,
    *,
    ra_shift_deg: float = 0.5,
    g_edges: np.ndarray | None = None,
    **pair_kwargs: Any,
) -> tuple[float, Table]:
    """Shifted-field chance-alignment estimate.

    Returns (global_rate, per-bin table with columns g_N_lo/hi, n_real, n_shift,
    r_chance). Pair kwargs are forwarded to `select_wide_pairs` for both the
    real and shifted catalogs.
    """
    g_edges = DEFAULT_G_EDGES if g_edges is None else np.asarray(g_edges, dtype=float)
    real = select_wide_pairs(stars, ra_shift_deg=0.0, **pair_kwargs)
    shifted = select_wide_pairs(stars, ra_shift_deg=ra_shift_deg, **pair_kwargs)
    n_real = max(len(real), 0)
    n_shift = len(shifted)
    global_rate = (n_shift / n_real) if n_real else float("nan")

    rows = []
    g_real = _as_float(real, "g_N_ms2") if n_real else np.array([])
    g_shift = _as_float(shifted, "g_N_ms2") if n_shift else np.array([])
    for lo, hi in zip(g_edges[:-1], g_edges[1:], strict=True):
        nr = int(np.sum((g_real >= lo) & (g_real < hi))) if n_real else 0
        ns = int(np.sum((g_shift >= lo) & (g_shift < hi))) if n_shift else 0
        r = (ns / nr) if nr else float("nan")
        rows.append((float(lo), float(hi), nr, ns, r))
    per_bin = Table(
        rows=rows or None,
        names=("g_N_lo_ms2", "g_N_hi_ms2", "n_real", "n_shift", "r_chance"),
        dtype=(float, float, int, int, float),
    )
    per_bin.meta["chance_alignment"] = {
        "method": "shifted-field",
        "ra_shift_deg": ra_shift_deg,
        "n_real": n_real,
        "n_shift": n_shift,
        "global_rate": global_rate,
    }
    return global_rate, per_bin


# ---------------------------------------------------------------------------
# Newtonian Monte Carlo (forward model)
# ---------------------------------------------------------------------------
def _sample_eccentricity(n: int, prior: EccPrior, rng: np.random.Generator) -> np.ndarray:
    if prior == "thermal":
        # CDF: e^2 = u  => e = sqrt(u)
        return np.sqrt(rng.random(n))
    if prior == "flat":
        return rng.random(n)
    raise ValueError(f"unknown eccentricity prior {prior!r}")


def newtonian_mock_pairs(
    n: int = 5000,
    *,
    ecc_prior: EccPrior = BASELINE_ECC_PRIOR,
    s_min_kau: float = BASELINE_CUTS["s_min_kau"],
    s_max_kau: float = BASELINE_CUTS["s_max_kau"],
    m_min: float = 0.2,
    m_max: float = 1.5,
    distance_pc: float = 100.0,
    seed: int = 42,
) -> Table:
    """Forward-model Newtonian wide binaries with projection effects.

    Samples true semi-major axes so that *projected* separations land in
    [s_min, s_max], draws masses, eccentricity from `ecc_prior`, isotropic
    orientation, and random phase; returns the same observable columns as
    `select_wide_pairs` (synthetic source_ids). No selection on measured PM
    error — the mock is the dynamical null for comparison after the same
    projected-s and g_N binning.
    """
    rng = np.random.default_rng(seed)
    # Sample true a log-uniform; project; keep those with s in window (rejection).
    pairs: list[tuple[Any, ...]] = []
    attempts = 0
    max_attempts = n * 40
    while len(pairs) < n and attempts < max_attempts:
        batch = max(n - len(pairs), 256)
        attempts += batch
        # log-uniform true a over a range that projects into the window
        a_kau = np.exp(
            rng.uniform(np.log(s_min_kau * 0.3), np.log(s_max_kau * 3.0), size=batch)
        )
        m1 = rng.uniform(m_min, m_max, size=batch)
        m2 = rng.uniform(m_min, m_max, size=batch)
        m_tot = m1 + m2
        e = _sample_eccentricity(batch, ecc_prior, rng)
        # isotropic: cos i uniform [-1,1]; ω, Ω, f uniform
        cosi = rng.uniform(-1.0, 1.0, size=batch)
        sini = np.sqrt(1.0 - cosi**2)
        omega = rng.uniform(0.0, 2.0 * np.pi, size=batch)
        f = rng.uniform(0.0, 2.0 * np.pi, size=batch)  # true anomaly (approx uniform; ok for mock)
        # radial distance
        r_kau = a_kau * (1.0 - e**2) / (1.0 + e * np.cos(f))
        # Thiele-Innes style projected separation (sky plane)
        # x_sky ~ r (cos(ω+f) cosΩ - sin(ω+f) sinΩ cosi) etc; for sep only:
        # projected r_proj / r = sqrt(1 - sin^2(θ) sin^2 i) with true anomaly geometry:
        # Use: s = r * sqrt( cos^2(θ) + sin^2(θ) cosi^2 ) where θ = ω+f
        theta = omega + f
        s_over_r = np.sqrt(np.cos(theta) ** 2 + (np.sin(theta) * cosi) ** 2)
        s_kau = r_kau * s_over_r
        keep = (s_kau >= s_min_kau) & (s_kau <= s_max_kau) & np.isfinite(s_kau)
        if not np.any(keep):
            continue
        a_kau = a_kau[keep]
        m1 = m1[keep]
        m2 = m2[keep]
        m_tot = m_tot[keep]
        e = e[keep]
        r_kau = r_kau[keep]
        s_kau = s_kau[keep]
        cosi = cosi[keep]
        sini = sini[keep]
        theta = theta[keep]
        f = f[keep]
        # vis-viva speed at true r
        s_au = s_kau * 1000.0
        r_au = r_kau * 1000.0
        a_au = a_kau * 1000.0
        v_true = _V_CIRC_1MSUN_1AU_KMS * np.sqrt(m_tot * (2.0 / r_au - 1.0 / a_au))
        v_true = np.maximum(v_true, 0.0)
        # Project vis-viva speed onto the sky: |v_sky| = |v| sinψ with ψ the
        # angle between the relative-velocity vector and the line of sight,
        # plus a mild face-on attenuation via sini.
        mu_los = rng.uniform(-1.0, 1.0, size=len(s_kau))
        v_sky = v_true * np.sqrt(np.maximum(0.0, 1.0 - mu_los**2))
        v_sky = v_sky * (0.5 + 0.5 * sini)

        v_circ = _v_circ_kms(m_tot, s_au)
        g_n = _g_n_ms2(m_tot, s_au)
        vtilde = np.where(v_circ > 0, v_sky / v_circ, np.nan)
        d_pc = np.full(len(s_kau), distance_pc)
        th = s_au / d_pc  # arcsec
        plx = 1000.0 / d_pc

        for k in range(len(s_kau)):
            pairs.append(
                (
                    f"mock1_{len(pairs)}", f"mock2_{len(pairs)}",
                    0.0, 0.0, th[k] / 3600.0, 0.0,
                    float(plx[k]), float(d_pc[k]), float(th[k]),
                    float(s_kau[k]), float(s_au[k]),
                    float(m1[k]), float(m2[k]), float(m_tot[k]),
                    float(v_sky[k]), float(v_circ[k]), float(vtilde[k]), float(g_n[k]),
                    1.0, 1.0,
                )
            )
            if len(pairs) >= n:
                break

    out = _pairs_table_from_rows(pairs[:n])
    out.meta["newtonian_mock"] = {
        "n_requested": n,
        "n_returned": len(out),
        "ecc_prior": ecc_prior,
        "s_min_kau": s_min_kau,
        "s_max_kau": s_max_kau,
        "seed": seed,
        "model": (
            "Keplerian two-body; log-uniform a; isotropic orientation; "
            "projected s and sky-plane relative velocity; comparison null for ṽ(g_N)"
        ),
    }
    return out


def binned_vtilde(
    pairs: Table,
    mock: Table | None = None,
    *,
    g_edges: np.ndarray | None = None,
    chance_per_bin: Table | None = None,
    min_pairs: int = 5,
) -> Table:
    """Bin ṽ by g_N; optionally attach Newtonian mock percentiles and R_chance."""
    g_edges = DEFAULT_G_EDGES if g_edges is None else np.asarray(g_edges, dtype=float)
    if "vtilde" not in pairs.colnames or "g_N_ms2" not in pairs.colnames:
        raise ValueError("pairs table needs vtilde and g_N_ms2 columns")

    g = _as_float(pairs, "g_N_ms2")
    vt = _as_float(pairs, "vtilde")
    g_m = _as_float(mock, "g_N_ms2") if mock is not None and len(mock) else None
    vt_m = _as_float(mock, "vtilde") if mock is not None and len(mock) else None

    chance_map: dict[tuple[float, float], float] = {}
    if chance_per_bin is not None and len(chance_per_bin):
        for row in chance_per_bin:
            chance_map[(float(row["g_N_lo_ms2"]), float(row["g_N_hi_ms2"]))] = float(
                row["r_chance"]
            )

    rows = []
    for lo, hi in zip(g_edges[:-1], g_edges[1:], strict=True):
        sel = np.isfinite(g) & np.isfinite(vt) & (g >= lo) & (g < hi)
        n = int(sel.sum())
        if n < min_pairs:
            continue
        v = vt[sel]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        err = 1.4826 * mad / np.sqrt(n)
        if vt_m is not None and g_m is not None:
            sm = np.isfinite(g_m) & np.isfinite(vt_m) & (g_m >= lo) & (g_m < hi)
            if sm.sum() >= 5:
                vm = vt_m[sm]
                m_med = float(np.median(vm))
                m_lo = float(np.percentile(vm, 16))
                m_hi = float(np.percentile(vm, 84))
            else:
                m_med = m_lo = m_hi = float("nan")
        else:
            m_med = m_lo = m_hi = float("nan")
        r_ch = chance_map.get((float(lo), float(hi)), float("nan"))
        mid = float(np.sqrt(lo * hi))
        rows.append(
            (
                float(lo), float(hi), mid, n, med, err,
                m_med, m_lo, m_hi, r_ch, int(r_ch > 0.1) if np.isfinite(r_ch) else 0,
            )
        )

    out = Table(
        rows=rows or None,
        names=(
            "g_N_lo_ms2", "g_N_hi_ms2", "g_N_mid_ms2", "n_pairs",
            "vtilde_med", "vtilde_err",
            "vtilde_mock_med", "vtilde_mock_lo", "vtilde_mock_hi",
            "r_chance", "flag_high_chance",
        ),
        dtype=(
            float, float, float, int,
            float, float,
            float, float, float,
            float, int,
        ),
    )
    out.meta["binned_vtilde"] = {
        "g_edges": list(map(float, g_edges)),
        "min_pairs": min_pairs,
        "statistic": "median ṽ; err = 1.4826*MAD/sqrt(n)",
        "s_definition": "projected separation (not deprojected)",
        "pair_meta": pairs.meta.get("pair_cuts"),
        "mock_meta": mock.meta.get("newtonian_mock") if mock is not None else None,
    }
    return out


def sensitivity_table(
    stars: Table | None = None,
    *,
    pairs_cache: dict[str, Table] | None = None,
    ecc_priors: tuple[EccPrior, ...] = ("thermal", "flat"),
    ruwe_max_values: tuple[float, ...] = (1.4, 1.2),
    n_mock: int = 4000,
    g_edges: np.ndarray | None = None,
    seed: int = 42,
    max_stars: int | None = 8000,
) -> Table:
    """Tabulate ṽ summary vs contested modeling choices.

    Rows: (ecc_prior, ruwe_max) × summary over the low-g half of the range
    (g_N < 1.2e-10 m/s²) and the high-g half. If `stars` is given, pairs are
    re-selected at each RUWE cut; otherwise `pairs_cache[str(ruwe)]` must supply
    precomputed pair tables.
    """
    g_edges = DEFAULT_G_EDGES if g_edges is None else np.asarray(g_edges, dtype=float)
    g_boundary = 1.2e-10
    rows = []
    for ruwe_max in ruwe_max_values:
        if pairs_cache is not None and str(ruwe_max) in pairs_cache:
            pairs = pairs_cache[str(ruwe_max)]
        elif stars is not None:
            pairs = select_wide_pairs(stars, ruwe_max=ruwe_max, max_stars=max_stars)
        else:
            raise ValueError("provide stars or pairs_cache for sensitivity_table")
        g = _as_float(pairs, "g_N_ms2") if len(pairs) else np.array([])
        vt = _as_float(pairs, "vtilde") if len(pairs) else np.array([])
        for ecc in ecc_priors:
            mock = newtonian_mock_pairs(n_mock, ecc_prior=ecc, seed=seed)
            gm = _as_float(mock, "g_N_ms2")
            vtm = _as_float(mock, "vtilde")
            for regime, mask_fn in (
                ("low_g", lambda x: x < g_boundary),
                ("high_g", lambda x: x >= g_boundary),
            ):
                if len(vt):
                    m = np.isfinite(g) & np.isfinite(vt) & mask_fn(g)
                    n = int(m.sum())
                    med = float(np.median(vt[m])) if n else float("nan")
                    mad = float(np.median(np.abs(vt[m] - med))) if n else float("nan")
                    err = 1.4826 * mad / np.sqrt(n) if n else float("nan")
                else:
                    n, med, err = 0, float("nan"), float("nan")
                mm = np.isfinite(gm) & np.isfinite(vtm) & mask_fn(gm)
                mock_med = float(np.median(vtm[mm])) if mm.sum() else float("nan")
                mock_lo = float(np.percentile(vtm[mm], 16)) if mm.sum() else float("nan")
                mock_hi = float(np.percentile(vtm[mm], 84)) if mm.sum() else float("nan")
                rows.append(
                    (
                        ecc, float(ruwe_max), regime, n, med, err,
                        mock_med, mock_lo, mock_hi,
                    )
                )
    if not rows:
        out = Table()
        out["ecc_prior"] = np.array([], dtype="U16")
        out["ruwe_max"] = np.array([], dtype=float)
        out["regime"] = np.array([], dtype="U16")
        out["n_pairs"] = np.array([], dtype=int)
        out["vtilde_med"] = np.array([], dtype=float)
        out["vtilde_err"] = np.array([], dtype=float)
        out["vtilde_mock_med"] = np.array([], dtype=float)
        out["vtilde_mock_lo"] = np.array([], dtype=float)
        out["vtilde_mock_hi"] = np.array([], dtype=float)
    else:
        out = Table()
        out["ecc_prior"] = np.array([r[0] for r in rows], dtype="U16")
        out["ruwe_max"] = np.array([r[1] for r in rows], dtype=float)
        out["regime"] = np.array([r[2] for r in rows], dtype="U16")
        out["n_pairs"] = np.array([r[3] for r in rows], dtype=int)
        out["vtilde_med"] = np.array([r[4] for r in rows], dtype=float)
        out["vtilde_err"] = np.array([r[5] for r in rows], dtype=float)
        out["vtilde_mock_med"] = np.array([r[6] for r in rows], dtype=float)
        out["vtilde_mock_lo"] = np.array([r[7] for r in rows], dtype=float)
        out["vtilde_mock_hi"] = np.array([r[8] for r in rows], dtype=float)
    out.meta["sensitivity"] = {
        "g_boundary_ms2": g_boundary,
        "ecc_priors": list(ecc_priors),
        "ruwe_max_values": list(ruwe_max_values),
        "n_mock": n_mock,
        "seed": seed,
        "note": (
            "Contested choices only; baseline is thermal + ruwe_max=1.4. "
            "Does not adjudicate Chae vs Banik — reports pipeline response."
        ),
    }
    return out


def make_synthetic_star_field(
    n_pairs: int = 200,
    n_field: int = 500,
    *,
    seed: int = 7,
    distance_pc: float = 80.0,
    ecc_prior: EccPrior = "thermal",
) -> Table:
    """Gaia-shaped star table containing bound Newtonian pairs + field stars.

    Used for offline tests and for a TAP-unavailable fallback study run. Bound
    pairs are planted with consistent parallax/PM; field stars are random.
    """
    rng = np.random.default_rng(seed)
    mock = newtonian_mock_pairs(
        n_pairs, ecc_prior=ecc_prior, seed=seed, distance_pc=distance_pc
    )
    rows: dict[str, list[Any]] = {
        "source_id": [],
        "ra": [],
        "dec": [],
        "parallax": [],
        "parallax_error": [],
        "pmra": [],
        "pmdec": [],
        "pmra_error": [],
        "pmdec_error": [],
        "phot_g_mean_mag": [],
        "bp_rp": [],
        "ruwe": [],
        "radial_velocity": [],
        "radial_velocity_error": [],
    }
    sid = 1
    plx = 1000.0 / distance_pc
    for row in mock:
        # place primary randomly on a 10x10 deg patch
        ra0 = float(rng.uniform(100.0, 110.0))
        dec0 = float(rng.uniform(20.0, 30.0))
        th = float(row["theta_arcsec"])
        # secondary offset mostly in RA
        dra = (th / 3600.0) / max(np.cos(np.deg2rad(dec0)), 0.2)
        # proper motion: shared bulk + relative
        pm0_ra = float(rng.normal(10.0, 5.0))
        pm0_dec = float(rng.normal(-5.0, 5.0))
        # half of relative PM on each component, arbitrary PA
        dpm = float(row["dv_kms"]) / (_K_MU_D * distance_pc)
        pa = float(rng.uniform(0, 2 * np.pi))
        for k, sign in ((1, +0.5), (2, -0.5)):
            rows["source_id"].append(sid)
            sid += 1
            rows["ra"].append(ra0 + (dra if k == 2 else 0.0))
            rows["dec"].append(dec0)
            rows["parallax"].append(plx + float(rng.normal(0, 0.02)))
            rows["parallax_error"].append(0.02)
            rows["pmra"].append(pm0_ra + sign * dpm * np.cos(pa))
            rows["pmdec"].append(pm0_dec + sign * dpm * np.sin(pa))
            rows["pmra_error"].append(0.05)
            rows["pmdec_error"].append(0.05)
            # mags from masses (rough)
            m = float(row[f"m{k}_msun"])
            abs_g = float(np.interp(np.log10(m), _LOGM_KNOTS[::-1], _MG_KNOTS[::-1]))
            rows["phot_g_mean_mag"].append(abs_g + 5 * np.log10(distance_pc) - 5)
            rows["bp_rp"].append(1.0)
            rows["ruwe"].append(1.05)
            rows["radial_velocity"].append(float(rng.normal(0, 5)))
            rows["radial_velocity_error"].append(1.0)

    # field contaminants
    for _ in range(n_field):
        rows["source_id"].append(sid)
        sid += 1
        rows["ra"].append(float(rng.uniform(100.0, 110.0)))
        rows["dec"].append(float(rng.uniform(20.0, 30.0)))
        rows["parallax"].append(plx + float(rng.normal(0, 0.5)))
        rows["parallax_error"].append(0.05)
        rows["pmra"].append(float(rng.normal(0, 30)))
        rows["pmdec"].append(float(rng.normal(0, 30)))
        rows["pmra_error"].append(0.1)
        rows["pmdec_error"].append(0.1)
        rows["phot_g_mean_mag"].append(float(rng.uniform(8, 17)))
        rows["bp_rp"].append(float(rng.uniform(0.5, 2.0)))
        rows["ruwe"].append(float(rng.uniform(0.9, 1.3)))
        rows["radial_velocity"].append(float("nan"))
        rows["radial_velocity_error"].append(float("nan"))

    t = Table(rows)
    t.meta["synthetic_star_field"] = {
        "n_pairs_planted": n_pairs,
        "n_field": n_field,
        "distance_pc": distance_pc,
        "ecc_prior": ecc_prior,
        "seed": seed,
    }
    return t
