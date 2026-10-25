"""Channels4World: navigate, select, read and prepare the Channels4World dataset.

    import channels4world as c4w
    ds = c4w.open("/path/to/channels4world")
"""

__version__ = "1.0.0"

from ._errors import (C4WError, FormatError, InsufficientSpaceError, IntegrityError, NotDownloadedError,
                      NotPublishedError, UnknownIDError)
from . import recipe
from .recipe import RecipeError
from .dataset import (BaseStation, Dataset, Frequency, Link, MissingReport, Positions, Record, Scene, Selection,
                      Split, open)
from .prepare import Plan, extract, to_cache

__all__ = [
    "open", "extract", "to_cache", "Dataset", "Split", "Scene", "Frequency", "BaseStation", "Record", "Link",
    "Positions", "Selection", "MissingReport", "Plan", "C4WError", "UnknownIDError", "NotPublishedError",
    "NotDownloadedError", "IntegrityError", "FormatError", "InsufficientSpaceError", "RecipeError", "recipe",
    "__version__",
]
