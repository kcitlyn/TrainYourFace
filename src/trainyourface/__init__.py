"""TrainYourFace: offline face recognition that knows when it is being fooled.

The public API is three names, imported lazily so that `import trainyourface` stays
fast and does not pull ONNX Runtime into a process that only wanted the version:

    from trainyourface import LivenessDetector, FaceID, TrustedFace

`LivenessDetector` answers "is this a real face?". `FaceID` answers "who is this,
and only if they're real". Keeping those separate is the whole point — see
`trainyourface.api`.
"""

from typing import TYPE_CHECKING

__version__ = "0.2.0"

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
