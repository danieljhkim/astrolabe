"""Pure analysis functions over astropy Tables — no network, no I/O side effects.

Each function takes Table(s) in and returns a Table or a matplotlib Figure. They are
unit-tested against small fixture tables (SPEC §3).
"""

from __future__ import annotations

from .crossmatch import crossmatch
from .hr import hr_diagram

__all__ = ["crossmatch", "hr_diagram"]
