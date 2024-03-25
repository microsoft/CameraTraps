import importlib.metadata as importlib_metadata
try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from .data import *
from .models import *
from .utils import *