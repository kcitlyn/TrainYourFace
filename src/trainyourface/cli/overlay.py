"""Frame overlay rendering for the live viewer.

The old UI drew a green box and a name. That is actively misleading for this
project, because a green box around a printed photo of an enrolled user is a
*successful attack* being reported as a success. The overlay has to distinguish
three states, not two:

    TRUSTED   (green)  - known identity AND confirmed live
    SPOOF     (red)    - liveness rejected it; identity deliberately not shown
    UNKNOWN   (amber)  - live face, but nobody we have enrolled
    UNVERIFIED (grey)  - no liveness model loaded, so trust is unestablished

The amber/grey distinction is the one that's easy to skip and worth keeping:
"we checked and it isn't you" and "we never checked" are different claims, and
collapsing them into one colour is how a demo ends up implying protection it
doesn't have.

Drawing is pure OpenCV primitives — no GUI toolkit — so the same code renders to
a window, to a saved frame, or over a headless video file.
"""

from __future__ import annotations

import numpy as np

from trainyourface.core.contracts import FaceObservation

# BGR, since that's OpenCV's channel order.
GREEN = (80, 220, 100)
RED = (60, 60, 240)
AMBER = (40, 180, 250)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
DARK = (28, 28, 30)

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, inlined to keep cv2 import lazy.


def face_state(face: FaceObservation) -> tuple[str, tuple[int, int, int]]:
    """Classify a face into a display state and colour.

    Ordering matters: the spoof check comes first, so a matched identity can
    never override a failed liveness check in the UI.
    """
    if face.liveness is not None and not face.liveness.is_live:
        return "SPOOF", RED
    if face.identity is None or not face.identity.is_known:
        if face.liveness is None:
            return "UNVERIFIED", GREY
        return "UNKNOWN", AMBER
    if face.liveness is None:
        return "UNVERIFIED", GREY
    return "TRUSTED", GREEN


def _corner_box(frame, box, color, thickness: int = 2, length_frac: float = 0.22) -> None:
    """Draw corner brackets instead of a full rectangle.

    Brackets leave the face visible, which matters when the point of the demo is
    for a viewer to compare a real face against a photo of it side by side.
    """
    import cv2

    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    n = max(8, int(min(box.width, box.height) * length_frac))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * n, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * n), color, thickness, cv2.LINE_AA)


def _label(frame, text: str, org: tuple[int, int], color, scale: float = 0.5) -> None:
    """Text on a filled plate, so it stays readable over any background."""
    import cv2

    (tw, th), base = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 10, y + 2), DARK, -1)
    cv2.putText(frame, text, (x + 5, y - base), _FONT, scale, color, 1, cv2.LINE_AA)


def _meter(frame, org: tuple[int, int], width: int, value: float, threshold: float, color) -> None:
    """A horizontal bar for the spoof score, with the threshold marked.

    Shown rather than just printing the number because the interesting thing in a
    live demo is watching the score cross the line as a phone is raised into
    frame. A bare float doesn't convey "how close was that".
    """
    import cv2

    x, y = org
    h = 5
    cv2.rectangle(frame, (x, y), (x + width, y + h), (70, 70, 70), -1)
    fill = int(np.clip(value, 0.0, 1.0) * width)
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    tick = x + int(np.clip(threshold, 0.0, 1.0) * width)
    cv2.line(frame, (tick, y - 2), (tick, y + h + 2), WHITE, 1)


def draw_face(frame: np.ndarray, face: FaceObservation) -> np.ndarray:
    """Annotate one face in place."""
    state, color = face_state(face)
    box = face.box.clipped(frame.shape[1], frame.shape[0])
    _corner_box(frame, box, color)

    # Primary line: the verdict. Identity is appended only when the face is
    # trusted — see the module docstring on not confirming a target to an
    # attacker whose spoof was rejected.
    headline = state
    if state == "TRUSTED" and face.identity is not None:
        headline = f"{face.identity.display_name}  {face.identity.similarity:.2f}"
    _label(frame, headline, (box.x1, box.y1 - 6), color, scale=0.55)

    if face.liveness is not None:
        y = box.y2 + 16
        _meter(
            frame,
            (box.x1, y),
            max(60, box.width),
            face.liveness.spoof_probability,
            face.liveness.threshold,
            color,
        )
        _label(
            frame,
            f"spoof {face.liveness.spoof_probability:.3f} / thr {face.liveness.threshold:.3f}",
            (box.x1, y + 26),
            WHITE,
            scale=0.4,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    result,
    *,
    extra: list[str] | None = None,
    warn: str | None = None,
) -> np.ndarray:
    """Per-stage timings and any standing warning, top-left.

    Stage timings are on screen during the demo on purpose: this is a project
    about edge latency, and a recorded demo that shows detect/liveness/recognize
    split out is more informative than one showing a single fps counter.
    """

    lines = [
        f"{result.total_ms:5.1f} ms  {result.fps:4.1f} fps",
        f"  detect    {result.detect_ms:5.1f} ms",
    ]
    if result.liveness_ms:
        lines.append(f"  liveness  {result.liveness_ms:5.1f} ms")
    if result.recognize_ms:
        lines.append(f"  recognize {result.recognize_ms:5.1f} ms")
    lines.append(f"faces: {len(result.faces)}")
    if extra:
        lines.extend(extra)

    y = 22
    for line in lines:
        _label(frame, line, (10, y), WHITE, scale=0.45)
        y += 20

    if warn:
        h = frame.shape[0]
        _label(frame, warn, (10, h - 12), AMBER, scale=0.45)
    return frame


def draw_capture_hud(
    frame: np.ndarray,
    *,
    label: str,
    subject: str,
    session: str,
    captured: int,
    target: int,
    has_face: bool,
    instrument: str | None = None,
) -> np.ndarray:
    """HUD for `tyf capture` — a progress bar and what's being recorded.

    Shows the label being written to disk at all times. Mislabeled PAD data is
    the worst possible outcome of a capture session (it teaches the model the
    inverse of the truth and the metrics still look fine), and the usual cause is
    forgetting which mode you're in. Keeping it on screen makes that mistake
    harder to make.
    """
    import cv2

    color = GREEN if label == "bona_fide" else RED
    h, w = frame.shape[:2]

    _label(frame, f"RECORDING: {label.upper()}", (10, 26), color, scale=0.6)
    meta = f"subject={subject}  session={session}"
    if instrument:
        meta += f"  instrument={instrument}"
    _label(frame, meta, (10, 50), WHITE, scale=0.42)

    bar_w = w - 20
    frac = captured / target if target else 0.0
    cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 22), (70, 70, 70), -1)
    cv2.rectangle(frame, (10, h - 30), (10 + int(bar_w * min(1.0, frac)), h - 22), color, -1)
    _label(frame, f"{captured}/{target} frames", (10, h - 38), WHITE, scale=0.42)

    if not has_face:
        _label(frame, "no face detected - not recording", (10, 74), AMBER, scale=0.45)

    _label(frame, "SPACE hold to record   q quit", (w - 250, 26), WHITE, scale=0.4)
    return frame
