"""TrainYourFace: offline face recognition that knows when it is being fooled.

The public API is three names, imported lazily so that `import trainyourface` stays
fast and does not pull ONNX Runtime into a process that only wanted the version:

    from trainyourface import LivenessDetector, FaceID, TrustedFace

`LivenessDetector` answers "is this a real face?". `FaceID` answers "who is this,
and only if they're real". Keeping those separate is the whole point — see
`trainyourface.api`.
"""

from typing import TYPE_CHECKING

# Read from installed package metadata rather than hardcoded, so pyproject.toml is
# the single source of truth. Two hand-maintained copies drift: the release
# workflow only checks the tag against pyproject.toml, so a bump that missed this
# file would publish a wheel whose `pip show` version and `__version__` disagree —
# permanently, since a PyPI version can never be re-uploaded.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("trainyourface")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    # Not installed (e.g. a bare checkout on sys.path). A placeholder is correct
    # here: there is no distribution to read a version from, and guessing one
    # would be a claim about a package that isn't installed.
    __version__ = "0.0.0+unknown"

__all__ = ["LivenessDetector", "FaceID", "TrustedFace", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from trainyourface.api import FaceID, LivenessDetector, TrustedFace


def __getattr__(name: str):
    """Resolve the public API on first access.

    A plain top-level import would load ONNX Runtime and OpenCV on `import
    trainyourface`, which costs ~1s and is wasted for anything that only reads
    __version__ — including pip metadata checks and the CLI's own --help path.
    """
    if name in ("LivenessDetector", "FaceID", "TrustedFace"):
        from trainyourface import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Make the lazily-exported names visible to dir() and autocomplete.

    A module-level __getattr__ resolves the API but leaves it invisible to
    `dir(trainyourface)`, so an IDE or a REPL user can't discover the three
    public names without already knowing them. Listing __all__ restores that.
    """
    return sorted(__all__)
