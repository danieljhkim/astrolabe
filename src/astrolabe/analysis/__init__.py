"""Pure analysis functions over astropy Tables — no network, no I/O side effects.

Each function takes Table(s) in and returns a Table or a matplotlib Figure. They are
unit-tested against small fixture tables (SPEC §3).
"""

from __future__ import annotations

from .crossmatch import crossmatch
from .hr import hr_diagram
from .kinematics import galactocentric_kinematics, rotation_curve
from .rar import radial_acceleration_relation
from .residuals import ephemeris_residuals
from .wide_binaries import (
    binned_vtilde,
    chance_alignment_rate,
    newtonian_mock_pairs,
    select_wide_pairs,
    sensitivity_table,
)

__all__ = [
    "binned_vtilde",
    "chance_alignment_rate",
    "crossmatch",
    "ephemeris_residuals",
    "galactocentric_kinematics",
    "hr_diagram",
    "newtonian_mock_pairs",
    "radial_acceleration_relation",
    "rotation_curve",
    "select_wide_pairs",
    "sensitivity_table",
]
