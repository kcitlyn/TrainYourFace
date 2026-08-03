"""Shared model/pipeline construction for CLI commands.

Centralized so every command reports the same thing about what's loaded, and so
the "liveness model is missing" path is handled in exactly one place. The old
code constructed detectors inline in several files, which is how two call sites
end up with different preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import typer

from trainyourface.core.pipeline import FacePipeline
from trainyourface.core.store import EnrollmentStore


def load_pipeline(
    *,
    liveness_path: Path | str | None = None,
    use_liveness: bool = True,
    recognize: bool = True,
    store_path: Path | str | None = None,
    match_threshold: float | None = None,
    liveness_threshold: float | None = None,
    identify_spoofs: bool = False,
    quiet: bool = False,
) -> FacePipeline:
    """Build a pipeline, downloading detector/embedder weights on first use.

    A missing liveness model is a warning, not an error: recognition-only is a
    legitimate mode and is how a new user gets to a working demo before they've
    collected PAD data. But it warns loudly, because in that mode a printed photo
    passes and the UI must not imply otherwise.
    """
    from trainyourface.core.detect import FaceDetector
    from trainyourface.core.embed import FaceEmbedder

    def note(msg: str) -> None:
        if not quiet:
            typer.secho(msg, fg=typer.colors.BRIGHT_BLACK, err=True)

    note("loading detector...")
    detector = FaceDetector()

    embedder = None
    store = None
    if recognize:
        note("loading embedder...")
        embedder = FaceEmbedder()
        store = EnrollmentStore(store_path)
        store.load()

    liveness = None
    if use_liveness:
        from trainyourface.liveness.predict import (
            LivenessModel,
            LivenessUnavailable,
            default_model_path,
        )

        path = Path(liveness_path) if liveness_path else default_model_path()
        try:
            liveness = LivenessModel(path, threshold=liveness_threshold)
            note(f"liveness: {path.name} (threshold {liveness.threshold:.4f})")
        except LivenessUnavailable as exc:
            typer.secho(f"\nWARNING: {exc}\n", fg=typer.colors.YELLOW, err=True)
            typer.secho(
                "Running WITHOUT liveness. Recognition alone cannot tell a real face "
                "from a photo of one, so no face will be reported as trusted.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    return FacePipeline(
        detector=detector,
        embedder=embedder,
        liveness=liveness,
        store=store,
        match_threshold=match_threshold,
        identify_spoofs=identify_spoofs,
    )


def open_camera(index: int = 0, width: int = 1280, height: int = 720):
    """Open a webcam with a clear error if it isn't available.

    OpenCV returns a VideoCapture that fails on read() rather than raising on
    open, so the failure otherwise surfaces as a confusing empty-frame loop. On
    macOS the usual cause is a missing camera permission, which is worth naming.
    """
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise typer.BadParameter(
            f"could not open camera {index}. On macOS, grant camera access to your "
            "terminal in System Settings > Privacy & Security > Camera. Use --camera N "
            "to select a different device."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def require_gui() -> None:
    """Fail early if OpenCV has no GUI support.

    Core installs use opencv-python-headless (deliberately — it avoids dragging a
    GUI stack onto a Raspberry Pi). cv2.imshow then raises a long, opaque error
    deep in the loop, so it's checked up front with the fix named.
    """
    import cv2

    if not hasattr(cv2, "imshow"):
        raise typer.BadParameter(
            "this OpenCV build has no GUI support. Install the viewer extra:\n"
            "  pip install 'trainyourface[demo]'"
        )
    try:
        cv2.namedWindow("__tyf_probe__")
        cv2.destroyWindow("__tyf_probe__")
    except cv2.error as exc:
        raise typer.BadParameter(
            "OpenCV cannot create a window (headless build or no display).\n"
            "  pip install 'trainyourface[demo]'\n"
            f"  original error: {exc}"
        ) from exc
