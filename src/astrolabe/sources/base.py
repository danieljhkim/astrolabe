"""The Source protocol — the load-bearing abstraction of astrolabe.

Every data source (survey, ephemeris service, telescope capture) is an adapter that
implements `Source`: it takes a plain dict of query params and returns an astropy
`Table` with standardized units and columns. Nothing outside `sources/` may import a
provider SDK; adapters are the only place provider-specific code lives.

Adapters normalize to a small house set of columns so downstream store/analysis code
is source-agnostic (SPEC §7 open question — kept intentionally minimal):

    source_id : object identifier as given by the source (str)
    ra        : right ascension, degrees (float)
    dec       : declination, degrees (float)

Additional columns (photometry, parallax, ...) pass through untouched with their
native names and units.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from astropy.table import Table

# Normalized column names every adapter is expected to provide when the concept
# exists in the source. Downstream code keys off these.
STANDARD_COLUMNS: tuple[str, ...] = ("source_id", "ra", "dec")


@runtime_checkable
class Source(Protocol):
    """A celestial data source.

    Implementations are cheap, stateless handles; the network call happens in
    `query()`. `name` is the stable slug used for storage paths and CLI selection.
    `kind` is the dataset kind this source produces (`store.KINDS`) — it decides
    where fetched data lands under `data/processed/<kind>/`.
    """

    name: str
    kind: str

    def query(self, params: dict[str, Any]) -> Table:
        """Run a query and return an astropy Table.

        `params` is source-specific (a cone search, an ADQL string, a target name).
        The returned Table carries units in its column metadata where meaningful and
        includes the STANDARD_COLUMNS the source can supply.
        """
        ...


def ensure_standard_columns(table: Table, *, require: bool = False) -> Table:
    """Validate that a Table carries the standard columns.

    With `require=True`, raises if any standard column is missing — used by adapters
    after normalization to fail loudly rather than store a malformed catalog.
    """
    missing = [c for c in STANDARD_COLUMNS if c not in table.colnames]
    if missing and require:
        raise ValueError(
            f"table is missing standard columns {missing}; has {list(table.colnames)}"
        )
    return table
