"""Frame overlay rendering for the live viewer.

The old UI drew a green box and a name. That is actively misleading for this
project, because a green box around a printed photo of an enrolled user is a
*successful attack* being reported as a success. The overlay distinguishes four
states, not two:

    TRUSTED    (green)  - known identity AND confirmed live
    SPOOF      (red)    - liveness rejected it; identity deliberately not shown
    UNKNOWN    (amber)  - live face, but nobody we have enrolled
    UNVERIFIED (grey)   - no liveness model loaded, so trust is unestablished

The amber/grey distinction is the one that's easy to skip and worth keeping:
"we checked and it isn't you" and "we never checked" are different claims, and
collapsing them into one colour is how a demo ends up implying protection it
doesn't have.

DESIGN CONSTRAINT: EVERY NUMBER ON SCREEN IS ONE A VIEWER CAN CHECK
-------------------------------------------------------------------
The HUD shows the spoof score, the threshold it's compared against, where that
threshold came from, and the per-stage latency. That's deliberate. A demo that
shows only a verdict asks to be trusted; one that shows the score crossing the
threshold lets a viewer verify the verdict was earned. It also makes the recorded
demo self-documenting — the numbers in the README are visible in the video.

Primitives live in `theme.py`; this module is the layout and the state logic.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from trainyourface.cli.theme import (
    AMBER,
    BLUE,
    DIM,
    FONT_BOLD,
    GREEN,
    GREY,
    PANEL,
    RED,
    WHITE,
    corner_box,
    meter,
    panel,
    pill,
    sparkline,
    text,
    text_size,
)
from trainyourface.core.contracts import FaceObservation

# Re-exported: tests and older call sites import colours from here.
__all__ = [
    "AMBER",
    "GREEN",
    "GREY",
    "RED",
    "WHITE",
    "ScoreHistory",
    "draw_capture_hud",
    "draw_face",
    "draw_hud",
    "face_state",
    "state_hint",
]

# How many frames of spoof score to keep for the trace. ~4 seconds at 30fps,
# long enough to show an attack entering frame and short enough that the trace
# reflects the present rather than the whole session.
HISTORY = 120


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


def state_hint(state: str) -> str:
    """One line explaining what a state means, for the legend.

    Written for someone watching a recording with no context. "SPOOF" alone
    doesn't convey that the system did the right thing, which is the entire point
    of the demo.
    """
    return {
        "TRUSTED": "recognized and confirmed live",
        "SPOOF": "not a live face - rejected before recognition",
        "UNKNOWN": "live person, not enrolled",
        "UNVERIFIED": "no liveness model - a photo would pass",
    }.get(state, "")


class ScoreHistory:
    """Per-face rolling spoof scores, keyed by position.

    Keyed by a coarse grid cell of the box centre rather than by a tracker: there
    is no tracker in this pipeline, and adding one to draw a graph would be a real
    component with real failure modes for a cosmetic gain. Coarse binning keeps
    the trace attached to a roughly-stationary face and lets it reset when someone
    moves across the frame, which is honest about what it is.
    """

    def __init__(self, cell: int = 120) -> None:
        self.cell = cell
        self._series: dict[tuple[int, int], deque[float]] = {}
        self._seen: dict[tuple[int, int], int] = {}
        self._tick = 0

    def _key(self, box) -> tuple[int, int]:
        return ((box.x1 + box.x2) // 2 // self.cell, (box.y1 + box.y2) // 2 // self.cell)

    def push(self, box, value: float) -> deque[float]:
        self._tick += 1
        key = self._key(box)
        series = self._series.setdefault(key, deque(maxlen=HISTORY))
        series.append(float(value))
        self._seen[key] = self._tick
        # Drop traces for faces that have left, so a long session doesn't
        # accumulate a dict entry per grid cell ever visited.
        for k, last in list(self._seen.items()):
            if self._tick - last > HISTORY:
                self._series.pop(k, None)
                self._seen.pop(k, None)
        return series

    def get(self, box) -> deque[float]:
        return self._series.get(self._key(box), deque())


def draw_face(
    frame: np.ndarray,
    face: FaceObservation,
    *,
    history: ScoreHistory | None = None,
    show_trace: bool = True,
) -> np.ndarray:
    """Annotate one face in place."""
    state, color = face_state(face)
    box = face.box.clipped(frame.shape[1], frame.shape[0])
    corner_box(frame, box, color)

    # Primary badge: the verdict, always. Identity is a second badge and only for
    # trusted faces — see the module docstring on not confirming a target to an
    # attacker whose spoof was rejected.
    pill_h = 0
    if box.y1 > 26:
        w, pill_h = pill(frame, state, (box.x1, box.y1 - 26), color, scale=0.5)
        if state == "TRUSTED" and face.identity is not None:
            ident = f"{face.identity.display_name} {face.identity.similarity:.2f}"
            if box.x1 + w + 6 + text_size(ident, 0.45, 1, FONT_BOLD)[0] + 16 < frame.shape[1]:
                pill(frame, ident, (box.x1 + w + 6, box.y1 - 26), BLUE, scale=0.45, filled=False)
    else:
        # Face is at the top edge; put the badge inside the box rather than
        # clipping it off-screen.
        _, pill_h = pill(frame, state, (box.x1 + 4, box.y1 + 4), color, scale=0.5)

    if face.liveness is None:
        return frame

    lv = face.liveness
    y = box.y2 + 10
    bar_w = max(70, box.width)

    # Below the box if there's room, otherwise above it. A meter drawn off the
    # bottom of the frame is worse than a slightly unusual position.
    if y + 34 > frame.shape[0]:
        y = max(2, box.y1 - 26 - pill_h - 34)

    meter(frame, (box.x1, y), bar_w, lv.spoof_probability, lv.threshold, color)
    text(
        frame,
        f"spoof {lv.spoof_probability:.3f}   thr {lv.threshold:.3f}",
        (box.x1, y + 24),
        DIM,
        scale=0.4,
    )

    if show_trace and history is not None:
        series = history.get(box)
        if len(series) > 1:
            tw, th = 92, 34
            tx = min(box.x1 + bar_w + 10, frame.shape[1] - tw - 4)
            if tx > box.x1:
                sparkline(frame, (tx, y - 10), (tw, th), series, lv.threshold, color)
    return frame


def draw_hud(
    frame: np.ndarray,
    result,
    *,
    extra: list[str] | None = None,
    warn: str | None = None,
    keys: str | None = None,
    subtitle: str | None = None,
    legend: bool = False,
    paused: bool = False,
) -> np.ndarray:
    """Status panel, key hints, and any standing warning.

    Stage timings are on screen during the demo on purpose: this is a project
    about edge latency, and a recorded demo that splits out detect/liveness/
    recognize is more informative than one showing a single fps counter. It also
    makes the bottleneck self-evident — detection dominates, which is the useful
    thing to know before optimizing anything.
    """
    import cv2

    h, w = frame.shape[:2]

    rows: list[tuple[str, str, tuple[int, int, int]]] = [
        ("total", f"{result.total_ms:5.1f} ms   {result.fps:4.1f} fps", WHITE),
        ("detect", f"{result.detect_ms:5.1f} ms", DIM),
    ]
    if result.liveness_ms:
        rows.append(("liveness", f"{result.liveness_ms:5.1f} ms", DIM))
    if result.recognize_ms:
        rows.append(("recognize", f"{result.recognize_ms:5.1f} ms", DIM))

    n_trusted = sum(1 for f in result.faces if f.is_trustworthy)
    n_spoof = sum(1 for f in result.faces if f.liveness is not None and not f.liveness.is_live)
    faces_line = f"{len(result.faces)}"
    if n_trusted:
        faces_line += f"   {n_trusted} trusted"
    if n_spoof:
        faces_line += f"   {n_spoof} spoof"
    rows.append(("faces", faces_line, GREEN if n_trusted else (RED if n_spoof else DIM)))

    label_w = max(text_size(k, 0.4)[0] for k, _, _ in rows)
    value_w = max(text_size(v, 0.44)[0] for _, v, _ in rows)
    pad = 12
    body_h = len(rows) * 19
    head_h = 26 if subtitle else 8
    box_w = label_w + value_w + pad * 3
    box_h = head_h + body_h + pad

    panel(frame, (10, 10), (10 + box_w, 10 + box_h), PANEL, alpha=0.66)

    y = 10 + pad
    if subtitle:
        text(frame, subtitle, (10 + pad, y + 8), WHITE, scale=0.45, weight=1, font=FONT_BOLD)
        y += 20
        cv2.line(frame, (10 + pad, y), (10 + box_w - pad, y), (60, 60, 66), 1)
        y += 4

    for key, value, color in rows:
        y += 19
        text(frame, key, (10 + pad, y), DIM, scale=0.4)
        text(frame, value, (10 + pad + label_w + pad, y), color, scale=0.44)

    if extra:
        ey = 10 + box_h + 16
        for line in extra:
            text(frame, line, (10 + pad, ey), WHITE, scale=0.44)
            ey += 19

    if legend:
        _draw_legend(frame)

    if keys:
        kw = text_size(keys, 0.4)[0]
        panel(frame, (w - kw - 26, 10), (w - 10, 36), PANEL, alpha=0.6)
        text(frame, keys, (w - kw - 18, 28), DIM, scale=0.4)

    if paused:
        # Unmissable, because a paused frame that looks live is confusing: the
        # timings freeze and it reads as a hang.
        pw, _ = text_size("PAUSED", 0.6, 1, FONT_BOLD)
        pill(frame, "PAUSED", ((w - pw - 16) // 2, 46), AMBER, scale=0.6)

    if warn:
        wt = text_size(warn, 0.44)[0]
        panel(frame, (10, h - 40), (min(w - 10, 10 + wt + 26), h - 10), PANEL, alpha=0.75)
        text(frame, warn, (22, h - 20), AMBER, scale=0.44)
    return frame


def _draw_legend(frame: np.ndarray) -> None:
    """Colour key, bottom-right. Toggleable, off by default.

    Worth having for a recorded demo where the viewer has no narration, but it's
    clutter once you know the states, so it's a keypress rather than permanent.
    """
    h, w = frame.shape[:2]
    entries = [
        (s, c, state_hint(s))
        for s, c in (
            ("TRUSTED", GREEN),
            ("SPOOF", RED),
            ("UNKNOWN", AMBER),
            ("UNVERIFIED", GREY),
        )
    ]

    row_h = 22
    name_w = max(text_size(s, 0.42, 1, FONT_BOLD)[0] for s, _, _ in entries) + 22
    hint_w = max(text_size(t, 0.38)[0] for _, _, t in entries)
    box_w = name_w + hint_w + 30
    box_h = len(entries) * row_h + 20
    x0, y0 = w - box_w - 10, h - box_h - 10

    panel(frame, (x0, y0), (x0 + box_w, y0 + box_h), PANEL, alpha=0.74)
    y = y0 + 10
    for name, color, hint in entries:
        y += row_h
        pill(frame, name, (x0 + 12, y - 17), color, scale=0.42)
        text(frame, hint, (x0 + name_w + 14, y - 3), DIM, scale=0.38)


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
    recording: bool = False,
    reason: str | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """HUD for `tyf capture` — progress and exactly what is being written.

    Shows the label being written to disk at all times. Mislabeled PAD data is the
    worst possible outcome of a capture session — it teaches the model the inverse
    of the truth and the metrics still look fine — and the usual cause is
    forgetting which mode you're in. Keeping it on screen makes that mistake
    harder to make.
    """
    import cv2

    h, w = frame.shape[:2]
    color = GREEN if label == "bona_fide" else RED

    # A red border while recording, so a paused capture is never mistaken for a
    # running one. Cheap, and the single most useful signal in this tool.
    if recording:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

    pw, ph = pill(frame, f"REC {label.upper()}" if recording else label.upper(), (12, 12), color)
    meta = f"{subject} / {session}" + (f" / {instrument}" if instrument else "")
    mw = text_size(meta, 0.42)[0]
    panel(
        frame,
        (12 + pw + 6, 12),
        (12 + pw + 6 + mw + 18, 12 + ph),
        PANEL,
        alpha=0.62,
        radius=ph // 2,
    )
    text(frame, meta, (12 + pw + 15, 12 + ph - 8), DIM, scale=0.42)

    # Progress: bar plus a count, since a bar alone can't tell you if you're at
    # 118 or 120 of a target and that's when you decide whether to keep going.
    frac = captured / target if target else 0.0
    bar_w = w - 24
    by = h - 30
    panel(frame, (12, by - 22), (12 + bar_w, by + 12), PANEL, alpha=0.66)
    # threshold=-1 suppresses the tick: `meter` marks a decision threshold, and a
    # progress bar has no decision point. Passing 1.0 drew a tick at the far right
    # that read as an unexplained mark on the bar.
    meter(frame, (12 + 10, by), bar_w - 20, min(1.0, frac), -1.0, color, height=8)
    text(frame, f"{captured} / {target} frames", (12 + 10, by - 6), WHITE, scale=0.42)

    # Right-aligned status, laid out right-to-left so the percentage and the fps
    # counter can't overlap. Both were previously positioned independently and
    # collided at some frame widths.
    right = 12 + bar_w - 10
    if fps is not None:
        f_txt = f"{fps:.0f} fps"
        right -= text_size(f_txt, 0.42)[0]
        text(frame, f_txt, (right, by - 6), DIM, scale=0.42)
        right -= 14
    pct = f"{frac * 100:.0f}%"
    text(frame, pct, (right - text_size(pct, 0.42)[0], by - 6), DIM, scale=0.42)

    # Why it isn't recording, stated plainly. "Nothing is happening" with no
    # explanation is the most frustrating possible state for a capture tool.
    blocker = reason or (None if has_face else "no face detected")
    if blocker:
        bw = text_size(blocker, 0.46)[0]
        panel(frame, ((w - bw) // 2 - 14, 52), ((w + bw) // 2 + 14, 84), PANEL, alpha=0.8)
        text(frame, blocker, ((w - bw) // 2, 74), AMBER, scale=0.46)
    elif not recording:
        hint = "hold SPACE to record"
        hw = text_size(hint, 0.46)[0]
        panel(frame, ((w - hw) // 2 - 14, 52), ((w + hw) // 2 + 14, 84), PANEL, alpha=0.7)
        text(frame, hint, ((w - hw) // 2, 74), WHITE, scale=0.46)

    keys = "SPACE record   q finish"
    kw = text_size(keys, 0.4)[0]
    panel(frame, (w - kw - 26, 12), (w - 12, 38), PANEL, alpha=0.6)
    text(frame, keys, (w - kw - 18, 30), DIM, scale=0.4)
    return frame
