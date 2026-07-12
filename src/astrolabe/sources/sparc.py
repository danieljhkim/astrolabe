"""SPARC adapter — Spitzer Photometry and Accurate Rotation Curves.

Lelli, McGaugh & Schombert 2016 (AJ 152, 157): 175 late-type galaxies with
3.6 μm surface photometry and HI/Hα rotation curves, published as two
machine-readable tables on the SPARC site. This is a whole-survey download,
not a cone search — the query selects which table:

    {"table": "rotation_curves"}   # default: per-radius mass models (Table 2)
    {"table": "galaxies"}          # per-galaxy properties (Table 1)

SPARC publishes no per-object sky coordinates, so (like the horizons vectors
branch) the normalized tables carry `source_id` (the SPARC galaxy name) but no
ra/dec. House columns use explicit units; the load-bearing conventions:

- `v_disk_kms` / `v_bul_kms` are the baryonic component curves at
  mass-to-light ratio Υ = 1 at 3.6 μm. Scale contributions by Υ (velocities by
  sqrt(Υ)); the standard SPARC values are Υ_disk = 0.5, Υ_bul = 0.7.
- `v_gas_kms` (and rarely `v_disk_kms`) can be *negative* where the enclosed
  surface density has a central depression: a component's contribution to
  g_bar is v·|v|/r, preserving the sign.
- `quality` is the SPARC quality flag Q: 1 high, 2 medium, 3 low (low = large
  asymmetries / bad inclination — excluded from standard analysis samples).

The download + MRT parse is isolated in `_run` for fixture-based testing.
"""

from __future__ import annotations

from typing import Any

from astropy.table import Table

_BASE_URL = "http://astroweb.cwru.edu/SPARC"
_URLS: dict[str, str] = {
    "galaxies": f"{_BASE_URL}/SPARC_Lelli2016c.mrt",
    "rotation_curves": f"{_BASE_URL}/MassModels_Lelli2016c.mrt",
}

# (published-name candidates, house name) — candidates tolerate the naming
# variants seen across the MRT header and its CDS/VizieR mirror.
_GALAXY_COLUMNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Galaxy", "ID", "Name"), "source_id"),
    (("T",), "hubble_type"),
    (("D", "Dist"), "dist_mpc"),
    (("e_D", "e_Dist"), "dist_err_mpc"),
    (("f_D", "f_Dist"), "dist_method"),
    (("Inc", "i"), "incl_deg"),
    (("e_Inc", "e_i"), "incl_err_deg"),
    (("L[3.6]", "L3.6", "L36"), "lum36_1e9lsun"),
    (("e_L[3.6]", "e_L3.6", "e_L36"), "lum36_err_1e9lsun"),
    (("Reff",), "r_eff_kpc"),
    (("SBeff",), "sb_eff_lsun_pc2"),
    (("Rdisk", "Rd"), "r_disk_kpc"),
    (("SBdisk", "SBdisk0"), "sb_disk_lsun_pc2"),
    (("MHI",), "mhi_1e9msun"),
    (("RHI",), "r_hi_kpc"),
    (("Vflat",), "v_flat_kms"),
    (("e_Vflat",), "v_flat_err_kms"),
    (("Q",), "quality"),
)
_GALAXY_REQUIRED = ("source_id", "dist_mpc", "incl_deg", "r_disk_kpc", "quality")

_ROTCURVE_COLUMNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ID", "Galaxy", "Name"), "source_id"),
    (("D", "Dist"), "dist_mpc"),
    (("R", "Rad"), "r_kpc"),
    (("Vobs",), "v_obs_kms"),
    (("e_Vobs",), "v_obs_err_kms"),
    (("Vgas",), "v_gas_kms"),
    (("Vdisk",), "v_disk_kms"),
    (("Vbul",), "v_bul_kms"),
    (("SBdisk",), "sb_disk_lsun_pc2"),
    (("SBbul",), "sb_bul_lsun_pc2"),
)
_ROTCURVE_REQUIRED = (
    "source_id", "r_kpc", "v_obs_kms", "v_obs_err_kms", "v_gas_kms", "v_disk_kms",
)

# Column semantics for delivery sidecars (ORB-10168) — kept next to the rename
# maps so the exchange contract and the normalization can't drift apart.
COLUMN_SEMANTICS: dict[str, dict[str, str]] = {
    "galaxies": {
        "source_id": "SPARC galaxy name (str)",
        "hubble_type": "numeric Hubble type T (0=S0 ... 11=BCD)",
        "dist_mpc": "distance, Mpc",
        "dist_err_mpc": "distance error, Mpc",
        "dist_method": "distance method flag (1=Hubble flow ... 5=RGB tip/Cepheids)",
        "incl_deg": "disk inclination, deg (90 = edge-on)",
        "incl_err_deg": "inclination error, deg",
        "lum36_1e9lsun": "total 3.6 um luminosity, 10^9 Lsun",
        "lum36_err_1e9lsun": "3.6 um luminosity error, 10^9 Lsun",
        "r_eff_kpc": "effective radius at 3.6 um, kpc",
        "sb_eff_lsun_pc2": "effective surface brightness at 3.6 um, Lsun/pc^2",
        "r_disk_kpc": "exponential disk scale length at 3.6 um, kpc",
        "sb_disk_lsun_pc2": "central disk surface brightness at 3.6 um, Lsun/pc^2",
        "mhi_1e9msun": "HI mass, 10^9 Msun",
        "r_hi_kpc": "HI radius at 1 Msun/pc^2, kpc",
        "v_flat_kms": "asymptotically flat rotation velocity, km/s (0 = not reached)",
        "v_flat_err_kms": "error on v_flat, km/s",
        "quality": "SPARC quality flag Q: 1 high, 2 medium, 3 low",
    },
    "rotation_curves": {
        "source_id": "SPARC galaxy name (str)",
        "dist_mpc": "assumed distance, Mpc (as in the galaxies table)",
        "r_kpc": "galactocentric radius, kpc",
        "v_obs_kms": "observed rotation velocity, km/s",
        "v_obs_err_kms": "error on v_obs, km/s",
        "v_gas_kms": "gas component velocity, km/s (may be negative: "
                     "contribution to g_bar is v*|v|/r)",
        "v_disk_kms": "stellar-disk component velocity at Upsilon[3.6]=1, km/s "
                      "(scale contribution by Upsilon_disk; standard 0.5)",
        "v_bul_kms": "bulge component velocity at Upsilon[3.6]=1, km/s "
                     "(scale contribution by Upsilon_bul; standard 0.7)",
        "sb_disk_lsun_pc2": "disk surface brightness at this radius, Lsun/pc^2",
        "sb_bul_lsun_pc2": "bulge surface brightness at this radius, Lsun/pc^2",
    },
}


class SPARCSource:
    """Adapter over the published SPARC machine-readable tables."""

    name = "sparc"
    kind = "catalog"

    def query(self, params: dict[str, Any]) -> Table:
        which = params.get("table", "rotation_curves")
        if which not in _URLS:
            raise ValueError(
                f"unknown SPARC table {which!r}; known: {', '.join(sorted(_URLS))}"
            )
        table = self._run(which)
        if which == "galaxies":
            return self._normalize(table, _GALAXY_COLUMNS, required=_GALAXY_REQUIRED)
        return self._normalize(table, _ROTCURVE_COLUMNS, required=_ROTCURVE_REQUIRED)

    # -- network seam (monkeypatched in tests) -----------------------------
    def _run(self, which: str) -> Table:  # pragma: no cover - network
        from urllib.request import urlopen

        from astropy.io import ascii as ascii_io

        with urlopen(_URLS[which], timeout=120) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        try:
            return ascii_io.read(text, format="mrt")
        except Exception:
            # The MRT reader is strict about the byte-by-byte header; the CDS
            # reader accepts the same files with a laxer parse.
            return ascii_io.read(text, format="cds")

    # -- normalization -----------------------------------------------------
    def _normalize(
        self,
        table: Table,
        columns: tuple[tuple[tuple[str, ...], str], ...],
        *,
        required: tuple[str, ...],
    ) -> Table:
        out = table.copy()
        for candidates, house in columns:
            for cand in candidates:
                if cand in out.colnames:
                    if cand != house:
                        out.rename_column(cand, house)
                    break
        missing = [c for c in required if c not in out.colnames]
        if missing:
            raise ValueError(
                f"SPARC table is missing columns {missing}; has {list(table.colnames)}"
            )
        out["source_id"] = [str(g).strip() for g in out["source_id"]]
        return out
