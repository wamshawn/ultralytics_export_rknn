import importlib.metadata
from .version import set_version, get_version
from .cli import main

try:
    __version__ = importlib.metadata.version(__name__)
    set_version(__version__)
except importlib.metadata.PackageNotFoundError:
    __version__ = get_version()


__all__ = ["main", "__version__"]
