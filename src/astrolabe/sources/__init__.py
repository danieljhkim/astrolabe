"""Source adapters. One module per provider; each exposes a `Source` implementation.

`get_source(name)` is the registry the CLI uses to resolve a source slug to an adapter
instance without importing every provider SDK eagerly (imports are lazy so a missing
optional provider doesn't break the whole package).
"""

from __future__ import annotations

from .base import STANDARD_COLUMNS, Source, ensure_standard_columns

__all__ = ["Source", "STANDARD_COLUMNS", "ensure_standard_columns", "get_source", "SOURCE_NAMES"]

# slug -> "module:class" (lazy import to keep provider SDKs optional).
_REGISTRY: dict[str, str] = {
    "gaia": "astrolabe.sources.gaia:GaiaSource",
    "sdss": "astrolabe.sources.sdss:SDSSSource",
    "horizons": "astrolabe.sources.horizons:HorizonsSource",
    "sparc": "astrolabe.sources.sparc:SPARCSource",
}

SOURCE_NAMES = tuple(_REGISTRY)


def get_source(name: str) -> Source:
    """Resolve a source slug to an adapter instance."""
    import importlib

    try:
        target = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown source {name!r}; known: {', '.join(SOURCE_NAMES)}"
        ) from None
    mod_name, cls_name = target.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()
