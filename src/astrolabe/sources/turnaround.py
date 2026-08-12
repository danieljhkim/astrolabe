"""Literature compilation of measured turnaround / zero-velocity radii (ORB-10754).

Offline curated table — no network. Each row is a published R0 / R_ta for a
nearby group or cluster, with the mass used by that analysis, an independent
mass where available, full citation, and an explicit circularity flag when the
mass was derived from the same R0 measurement.

This is a literature-compilation Source (kind=catalog), not a survey adapter.
query() ignores params except optional cosmology passthrough recorded in meta.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.table import Table

# Curated rows. Masses in 10^12 Msun. R_ta in Mpc.
# circularity_flag: True when mass_analysis is derived from the measured R0
# (R0 method / spherical-collapse inversion) — comparing R_meas to R_pred(M_R0)
# is then partly tautological; use mass_indep for a non-circular ratio.
_ROWS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "Local_Group",
        "system_name": "Local Group (MW+M31)",
        "r_ta_mpc": 0.91,
        "r_ta_err_mpc": 0.05,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.5,
        "mass_analysis_err_1e12msun": 0.2,
        "mass_analysis_provenance": (
            "R0 method under Planck (Ωm=0.315, H0=67.3): "
            "M/Msun = 1.95e12 (R0/Mpc)^3 → 1.5e12"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 2.9,
        "mass_indep_err_1e12msun": 0.3,
        "mass_indep_provenance": (
            "Orbital sum MW+M31 satellites (Kashibadze & Karachentsev 2018 "
            "Table 3 median / present-paper orbital: ~2.9e12)"
        ),
        "citation": (
            "Kashibadze O.G., Karachentsev I.D. 2018, A&A 609, A11 "
            "(Cosmic flow around local massive galaxies); "
            "updates Karachentsev & Kashibadze 2006"
        ),
        "bibcode": "2018A&A...609A..11K",
        "notes": (
            "Preferred modern LG R0; barycentre x=Dc/DM31~0.4–0.55. "
            "R0-mass is circular; orbital mass is independent but larger."
        ),
    },
    {
        "source_id": "Local_Group_KK06",
        "system_name": "Local Group (Karachentsev & Kashibadze 2006)",
        "r_ta_mpc": 0.96,
        "r_ta_err_mpc": 0.03,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.29,
        "mass_analysis_err_1e12msun": 0.14,
        "mass_analysis_provenance": (
            "R0 + EdS/empty-universe formula M=(π²/8G) R0³ T0^{-2}, T0=13.7 Gyr"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 1.9,
        "mass_indep_err_1e12msun": 0.2,
        "mass_indep_provenance": (
            "Same R0 with Ωm=0.24 flat-universe conversion quoted by "
            "Karachentsev et al. 2009 MNRAS 393, 1265 (still R0-based, "
            "cosmology-dependent; not virial)"
        ),
        "citation": (
            "Karachentsev I.D., Kashibadze O.G. 2006, Astrophysics 49, 3 "
            "(Total masses of the Local Group and M81 group...)"
        ),
        "bibcode": "2006Ap.....49....3K",
        "notes": (
            "Classic R0=0.96±0.03; mass_indep here is still R0-derived under "
            "different cosmology (flag remains circular for both mass columns "
            "w.r.t. velocity-field independence). Prefer Local_Group row."
        ),
    },
    {
        "source_id": "M81",
        "system_name": "M81/M82 group",
        "r_ta_mpc": 0.89,
        "r_ta_err_mpc": 0.05,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.03,
        "mass_analysis_err_1e12msun": 0.17,
        "mass_analysis_provenance": (
            "R0 + EdS/T0 formula (Karachentsev & Kashibadze 2006)"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 4.9,
        "mass_indep_err_1e12msun": 1.5,
        "mass_indep_provenance": (
            "Orbital mass log Morb/Msun=12.69 (Karachentsev & Kudrya 2014 AJ 148, 50)"
        ),
        "citation": (
            "Karachentsev I.D., Kashibadze O.G. 2006, Astrophysics 49, 3; "
            "orbital mass: Karachentsev & Kudrya 2014, AJ 148, 50"
        ),
        "bibcode": "2006Ap.....49....3K",
        "notes": (
            "Barycentre at x~0.35 toward M82. Large orbital vs R0 mass tension "
            "is a known Local Volume puzzle (see Kashibadze & Karachentsev 2018)."
        ),
    },
    {
        "source_id": "CenA",
        "system_name": "Centaurus A group",
        "r_ta_mpc": 1.40,
        "r_ta_err_mpc": 0.15,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 5.4,
        "mass_analysis_err_1e12msun": 1.8,
        "mass_analysis_provenance": (
            "R0≈1.4 Mpc zero-velocity surface; mass via LCDM R0-M "
            "(~1.95e12 R0³) at R0=1.4 → ~5.4e12"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 7.8,
        "mass_indep_err_1e12msun": 2.5,
        "mass_indep_provenance": (
            "Orbital mass log Morb/Msun=12.89 (Karachentsev & Kudrya 2014)"
        ),
        "citation": (
            "Karachentsev I.D. et al. 2007, AJ 133, 504 "
            "(The Centaurus Group...); R0~1.4 Mpc also Karachentsev et al. 2006 AJ; "
            "complex summary Karachentsev 2005, AJ 129, 178"
        ),
        "bibcode": "2007AJ....133..504K",
        "notes": (
            "CenA/M83 is a binary complex; R0 quoted for CenA subgroup. "
            "R0 uncertainty enlarged vs some single-number quotes."
        ),
    },
    {
        "source_id": "IC342_Maffei",
        "system_name": "IC342/Maffei complex",
        "r_ta_mpc": 0.90,
        "r_ta_err_mpc": 0.15,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.4,
        "mass_analysis_err_1e12msun": 0.7,
        "mass_analysis_provenance": (
            "R0 in 0.9–1.3 Mpc range for Local Volume groups "
            "(Karachentsev 2005 summary); adopt 0.9 with broad err; "
            "M from LCDM R0-M"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 3.2,
        "mass_indep_err_1e12msun": 1.2,
        "mass_indep_provenance": (
            "Orbital masses IC342 (log 12.51) + Maffei2 (comparable) "
            "order ~ few × 10^12 (Karachentsev & Kudrya 2014 Table)"
        ),
        "citation": (
            "Karachentsev I.D. et al. 2003, A&A 408, 111 "
            "(Distances to nearby galaxies around IC 342); "
            "Karachentsev 2005, AJ 129, 178"
        ),
        "bibcode": "2003A&A...408..111K",
        "notes": (
            "Zone of Avoidance; distance/R0 quality lower than LG/M81. "
            "Treat as indicative."
        ),
    },
    {
        "source_id": "NGC253_Sculptor",
        "system_name": "NGC 253 / Sculptor filament",
        "r_ta_mpc": 0.70,
        "r_ta_err_mpc": 0.20,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 0.7,
        "mass_analysis_err_1e12msun": 0.5,
        "mass_analysis_provenance": (
            "R0 method on filament core; system likely unrelaxed "
            "(long crossing time) so R0-mass is unreliable"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 1.5,
        "mass_indep_err_1e12msun": 0.8,
        "mass_indep_provenance": (
            "Orbital/virial estimates ~1.5e12 (Karachentsev et al. 2003; "
            "Karachentsev & Kudrya 2014 log Morb~12.18)"
        ),
        "citation": (
            "Karachentsev I.D. et al. 2003, A&A 404, 93 "
            "(Distances to nearby galaxies in Sculptor...); "
            "Karachentsev 2005, AJ 129, 178"
        ),
        "bibcode": "2003A&A...404...93K",
        "notes": (
            "Expanding filament, not a virialized group — Karachentsev 2005 "
            "flags R0-mass as 3–8× below virial. Heterogeneity caveat."
        ),
    },
    {
        "source_id": "NGC4736_CVn",
        "system_name": "NGC 4736 / Canes Venatici I cloud",
        "r_ta_mpc": 0.90,
        "r_ta_err_mpc": 0.20,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.4,
        "mass_analysis_err_1e12msun": 0.9,
        "mass_analysis_provenance": (
            "R0 ~0.9±0.1 Mpc for CVn complex (Peirani & de Freitas Pacheco "
            "2008 citing Karachentsev et al. 2003a); M from LCDM R0-M"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 2.7,
        "mass_indep_err_1e12msun": 1.0,
        "mass_indep_provenance": (
            "Orbital mass log Morb/Msun=12.43 (Karachentsev & Kudrya 2014)"
        ),
        "citation": (
            "Karachentsev I.D. et al. 2003, A&A 398, 467 "
            "(Galaxy flow in the Canes Venatici I cloud); "
            "Peirani & de Freitas Pacheco 2008, A&A 488, 845"
        ),
        "bibcode": "2003A&A...398..467K",
        "notes": (
            "Loose expanding cloud; crossing time ~ Hubble time. "
            "R0-based mass not trusted as dynamical equilibrium mass."
        ),
    },
    {
        "source_id": "Virgo",
        "system_name": "Virgo cluster",
        "r_ta_mpc": 7.2,
        "r_ta_err_mpc": 0.7,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 800.0,
        "mass_analysis_err_1e12msun": 230.0,
        "mass_analysis_provenance": (
            "R0 method: M = (8.0±2.3)e14 Msun from R0=(7.2±0.7) Mpc "
            "(Karachentsev et al. 2014)"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 630.0,
        "mass_indep_err_1e12msun": 90.0,
        "mass_indep_provenance": (
            "Virial mass (6.3±0.9)e14 Msun from 1D profile "
            "(Kashibadze, Karachentsev & Nasonova 2020, A&A 635, A135)"
        ),
        "citation": (
            "Karachentsev I.D., Tully R.B., Wu P.-F., Shaya E.J., Dolphin A.E. "
            "2014, ApJ 782, 4 (Infall of Nearby Galaxies into the Virgo Cluster...); "
            "virial: Kashibadze et al. 2020, A&A 635, A135"
        ),
        "bibcode": "2014ApJ...782....4K",
        "notes": (
            "Cluster-scale zero-velocity surface. R0 later refined to 7.0–7.3 Mpc "
            "(Kashibadze et al. 2020). Primary cluster row for theory comparison."
        ),
    },
    {
        "source_id": "Fornax_Eridanus",
        "system_name": "Fornax–Eridanus complex",
        "r_ta_mpc": 4.60,
        "r_ta_err_mpc": 1.10,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 190.0,
        "mass_analysis_err_1e12msun": 120.0,
        "mass_analysis_provenance": (
            "R0=4.60 Mpc with CI [3.38, ~5.6] (Nasonova et al. 2011); "
            "M via LCDM R0-M (~1.95e12 R0³) → ~1.9e14; err from R0 CI"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": np.nan,
        "mass_indep_err_1e12msun": np.nan,
        "mass_indep_provenance": (
            "No single homogeneous independent mass out to R0 adopted; "
            "caustic masses exist on smaller radii only"
        ),
        "citation": (
            "Nasonova O.G., de Freitas Pacheco J.A., Karachentsev I.D. 2011, "
            "A&A 532, A104 (Hubble flow around Fornax cluster of galaxies)"
        ),
        "bibcode": "2011A&A...532A.104N",
        "notes": (
            "Wide R0 CI — ratio uncertainties dominate. Complex includes "
            "Eridanus cloud."
        ),
    },
    {
        "source_id": "LV_synthetic_stack",
        "system_name": "Local Volume synthetic stacked group (14 groups)",
        "r_ta_mpc": 0.93,
        "r_ta_err_mpc": 0.02,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.6,
        "mass_analysis_err_1e12msun": 0.2,
        "mass_analysis_provenance": (
            "Stacked R0 under Planck → M~(1.6±0.2)e12 "
            "(Kashibadze & Karachentsev 2018)"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 2.6,
        "mass_indep_err_1e12msun": 0.5,
        "mass_indep_provenance": (
            "Mean orbital mass of stacked hosts log Morb~12.42 "
            "(~2.6e12; same paper)"
        ),
        "citation": (
            "Kashibadze O.G., Karachentsev I.D. 2018, A&A 609, A11 "
            "(synthetic group, barycentre minor-attractor fit)"
        ),
        "bibcode": "2018A&A...609A..11K",
        "notes": (
            "Not a single physical object — mean Local Volume group scale. "
            "Useful as coverage-honest average of the Karachentsev program."
        ),
    },
    {
        "source_id": "M83",
        "system_name": "M83 group",
        "r_ta_mpc": 1.0,
        "r_ta_err_mpc": 0.2,
        "observable_kind": "zero_velocity_surface",
        "mass_analysis_1e12msun": 1.95,
        "mass_analysis_err_1e12msun": 1.0,
        "mass_analysis_provenance": (
            "R0 order ~1 Mpc (Karachentsev 2005 group summary range 0.9–1.3); "
            "M from LCDM R0-M at R0=1.0"
        ),
        "circularity_flag": True,
        "mass_indep_1e12msun": 1.0,
        "mass_indep_err_1e12msun": 0.4,
        "mass_indep_provenance": (
            "Virial/orbital ~1e12 for M83 subgroup (Karachentsev 2005 Table 10)"
        ),
        "citation": (
            "Karachentsev I.D. et al. 2002, A&A 385, 21 "
            "(New distances to galaxies in the Centaurus A group); "
            "Karachentsev 2005, AJ 129, 178"
        ),
        "bibcode": "2002A&A...385...21K",
        "notes": (
            "Paired with CenA; R0 less tightly constrained than LG/M81. "
            "Indicative row."
        ),
    },
    )

COLUMN_SEMANTICS: dict[str, str] = {
    "source_id": "stable system slug (str)",
    "system_name": "human-readable system name",
    "r_ta_mpc": "measured turnaround / zero-velocity radius, Mpc",
    "r_ta_err_mpc": "uncertainty on r_ta, Mpc",
    "observable_kind": (
        "zero_velocity_surface | turnaround | theory_bound"
    ),
    "mass_analysis_1e12msun": (
        "mass used by the cited analysis, 10^12 Msun"
    ),
    "mass_analysis_err_1e12msun": "uncertainty on mass_analysis, 10^12 Msun",
    "mass_analysis_provenance": "how mass_analysis was obtained",
    "circularity_flag": (
        "True if mass_analysis derives from the same R0/turnaround analysis"
    ),
    "mass_indep_1e12msun": (
        "independent mass (orbital/virial/X-ray/etc.) when available, 10^12 Msun"
    ),
    "mass_indep_err_1e12msun": "uncertainty on mass_indep, 10^12 Msun",
    "mass_indep_provenance": "how mass_indep was obtained",
    "citation": "full bibliographic citation",
    "bibcode": "ADS bibcode when known",
    "notes": "heterogeneity / quality caveats",
}


class TurnaroundSource:
    """Curated literature table of measured turnaround / zero-velocity radii."""

    name = "turnaround"
    kind = "catalog"

    def query(self, params: dict[str, Any] | None = None) -> Table:
        """Return the literature compilation as an astropy Table.

        params are accepted for Source-protocol uniformity and recorded in
        meta; the table is offline and params do not filter rows.
        """
        params = dict(params or {})
        col_names = list(_ROWS[0].keys())
        data = {c: [r[c] for r in _ROWS] for c in col_names}
        # Force float columns that may mix nan
        for c in (
            "r_ta_mpc",
            "r_ta_err_mpc",
            "mass_analysis_1e12msun",
            "mass_analysis_err_1e12msun",
            "mass_indep_1e12msun",
            "mass_indep_err_1e12msun",
        ):
            data[c] = np.asarray(data[c], dtype=float)
        data["circularity_flag"] = np.asarray(data["circularity_flag"], dtype=bool)
        table = Table(data)
        # No sky coordinates in this compilation (system-level, not object-level).
        table.meta["turnaround_query"] = {
            "n_rows": len(table),
            "params": params,
            "task": "ORB-10754",
            "discipline": (
                "literature compilation; every mass_analysis row carries "
                "citation + circularity_flag; independent masses preferred "
                "for non-circular R_meas/R_pred ratios"
            ),
            "lcdm_bound_reference": (
                "Pavlidou & Tomaras 2014, JCAP 09, 020 (arXiv:1310.1920): "
                "R_ta,max = (3GM/Λc²)^{1/3}; used for r_pt_max columns in analysis"
            ),
        }
        return table


def literature_rows() -> tuple[dict[str, Any], ...]:
    """Expose curated rows for tests / notebooks."""
    return _ROWS
