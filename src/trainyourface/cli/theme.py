"""Shared visual language for the on-frame overlay.

Split out from `overlay.py` so the drawing primitives can be reused by the
viewer, the capture tool, and anything that renders to a file, and so the colours
are defined once. Two call sites picking slightly different greens is a small
thing that makes a demo look unfinished.

Everything here draws with OpenCV primitives onto a numpy array — no GUI toolkit.
That keeps the same code path working for a live window, a saved screenshot, and
a headless render over a video file, which is what makes a README GIF
reproducible instead of hand-recorded.

`cv2` is imported inside functions, not at module scope, because this module is
imported by the CLI's help text path and a core install on a Raspberry Pi
shouldn't pay for an OpenCV import to print `--help`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# BGR, since that's OpenCV's channel order. Deliberately desaturated relative to
# pure (0,255,0)-style colours: full-saturation overlays on a camera feed read as
# harsh and make the underlying face harder to judge, which matters when the whole
# point is comparing a real face against a photo of it.
GREEN = (90, 214, 108)
RED = (68, 68, 240)
AMBER = (48, 176, 250)
GREY = (156, 156, 156)
BLUE = (232, 168, 88)
WHITE = (248, 248, 248)
DIM = (168, 168, 172)
PANEL = (26, 26, 30)
TRACK = (72, 72, 78)

# cv2 font constants, inlined so the import above can stay lazy.
FONT = 0  # FONT_HERSHEY_SIMPLEX
FONT_BOLD = 2  # FONT_HERSHEY_DUPLEX — heavier stroke, used for headings only.


def text_size(s: str, scale: float, weight: int = 1, font: int = FONT) -> tuple[int, int]:
    """(width, height) including the baseline drop, so callers can lay out rows."""
    import cv2

    (w, h), base = cv2.getTextSize(s, font, scale, weight)
    return w, h + base


def text(
    frame: np.ndarray,
    s: str,
    org: tuple[int, int],
    color=WHITE,
    scale: float = 0.45,
    weight: int = 1,
    font: int = FONT,
    shadow: bool = True,
) -> None:
    """Draw text at a bottom-left origin, with a shadow for legibility.

    The shadow is what lets text sit directly on a camera frame without a plate
    behind it. A frame can be any colour at any pixel, so unshadowed text is
    readable in a test and invisible in a bright room.

    The shadow uses the SAME stroke weight as the text, offset by one pixel. A
    heavier stroke would seem like the obvious way to get a stronger shadow, but
    OpenCV renders thicker strokes wider — 68px vs 64px for this font at 0.38 —
    so the shadow overhangs the glyphs it's meant to sit behind and reads as a
    dark duplicate letter trailing every string. `text_size` also reports the
    thickness-1 width, so a heavier shadow silently overflows any panel sized
    from it.
    """
    import cv2

    x, y = org
    if shadow:
        cv2.putText(frame, s, (x + 1, y + 1), font, scale, (0, 0, 0), weight, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), font, scale, color, weight, cv2.LINE_AA)


def panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color=PANEL,
    alpha: float = 0.62,
    radius: int = 8,
    border=None,
) -> None:
    """A translucent rounded panel, blended in place.

    Translucent rather than opaque so the HUD never fully hides part of the scene
    — during a demo the interesting thing may be happening behind the panel.
    Clipped to the frame because callers compute positions from face boxes, which
    routinely run off-frame.
    """
    h, w = frame.shape[:2]
    x1 = int(np.clip(top_left[0], 0, w - 1))
    y1 = int(np.clip(top_left[1], 0, h - 1))
    x2 = int(np.clip(bottom_right[0], 0, w))
    y2 = int(np.clip(bottom_right[1], 0, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return

    region = frame[y1:y2, x1:x2]
    r = int(min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    # The coverage mask is built from GEOMETRY, in its own single-channel buffer,
    # not by testing the colour buffer for non-zero pixels. Deriving it from
    # colour meant a black panel had an all-zero mask and silently drew nothing —
    # so `panel(..., (0, 0, 0))` was a no-op, which is a plausible thing for a
    # caller to ask for and gave no hint as to why it didn't work.
    mask = np.zeros(region.shape[:2], np.uint8)
    _filled_rounded(mask, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), 255, r)

    shape = np.empty_like(region)
    shape[:] = color
    blended = (region * (1 - alpha) + shape * alpha).astype(np.uint8)
    # Blend only inside the rounded shape, so the corners don't pick up a faint
    # rectangular halo.
    region[:] = np.where(mask[..., None].astype(bool), blended, region)

    if border is not None:
        _rounded_outline(frame, (x1, y1), (x2 - 1, y2 - 1), border, r, 1)


def _filled_rounded(img, tl, br, color, r: int) -> None:
    import cv2

    x1, y1 = tl
    x2, y2 = br
    if r <= 0:
        cv2.rectangle(img, tl, br, color, -1)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1)


def _rounded_outline(img, tl, br, color, r: int, thickness: int) -> None:
    import cv2

    x1, y1 = tl
    x2, y2 = br
    if r <= 0:
        cv2.rectangle(img, tl, br, color, thickness, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for (cx, cy), a in (
        ((x1 + r, y1 + r), 180),
        ((x2 - r, y1 + r), 270),
        ((x2 - r, y2 - r), 0),
        ((x1 + r, y2 - r), 90),
    ):
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)


def pill(
    frame: np.ndarray,
    s: str,
    org: tuple[int, int],
    color,
    scale: float = 0.5,
    weight: int = 1,
    filled: bool = True,
) -> tuple[int, int]:
    """A badge with the state colour. Returns its (width, height).

    Filled for the primary verdict and outlined for secondary information, so the
    verdict is the thing the eye lands on first without needing to read anything.
    """
    tw, th = text_size(s, scale, weight, FONT_BOLD)
    pad_x, pad_y = 8, 5
    x, y = org
    w, h = tw + pad_x * 2, th + pad_y * 2

    if filled:
        panel(frame, (x, y), (x + w, y + h), color, alpha=0.92, radius=h // 2)
        # Dark text on a saturated fill beats white text: it survives being
        # screenshotted, scaled down for a README, and re-compressed.
        text(
            frame, s, (x + pad_x, y + h - pad_y - 3), (18, 18, 20), scale, weight, FONT_BOLD, False
        )
    else:
        panel(frame, (x, y), (x + w, y + h), PANEL, alpha=0.6, radius=h // 2, border=color)
        text(frame, s, (x + pad_x, y + h - pad_y - 3), color, scale, weight, FONT_BOLD, False)
    return w, h


def meter(
    frame: np.ndarray,
    org: tuple[int, int],
    width: int,
    value: float,
    threshold: float,
    color,
    height: int = 6,
) -> None:
    """A horizontal bar with the decision threshold marked.

    Shown rather than only printing the number because the interesting moment in a
    live demo is watching the score cross the line as a phone is raised into
    frame. A bare float doesn't convey "how close was that".

    A negative `threshold` suppresses the tick, for callers reusing this as a plain
    progress bar. A progress bar has no decision point, and a tick on one reads as
    an unexplained mark the viewer has to account for.
    """
    import cv2

    x, y = org
    r = height // 2
    _filled_rounded(frame, (x, y), (x + width, y + height), TRACK, r)
    fill = int(np.clip(value, 0.0, 1.0) * width)
    if fill > r:
        _filled_rounded(frame, (x, y), (x + fill, y + height), color, r)

    if threshold >= 0.0:
        tick = x + int(np.clip(threshold, 0.0, 1.0) * width)
        cv2.line(frame, (tick, y - 3), (tick, y + height + 3), WHITE, 1, cv2.LINE_AA)


def sparkline(
    frame: np.ndarray,
    org: tuple[int, int],
    size: tuple[int, int],
    values: deque[float] | list[float],
    threshold: float,
    color,
) -> None:
    """Plot recent spoof scores against the threshold.

    This is the most useful single element in the viewer. A momentary score tells
    you the current verdict; the trace tells you whether the model is confident
    and stable or oscillating across the line, and it makes the moment an attack
    enters frame legible in a recording. y is inverted so "up" means "more likely
    a spoof", matching the meter.

    The plot area is inset from the panel. Mapping 0..1 onto the full panel height
    looks correct until you use it: a confident model sits at ~0.02 or ~0.98
    almost all the time, which is exactly where the trace lands on the border and
    becomes unreadable. Since confident is the normal case, the inset is what
    makes the graph legible rather than a refinement.

    The inset is sized from the endpoint marker's radius, not picked by eye — the
    marker is the widest thing drawn, so anything smaller lets it hang over the
    edge at v=0 or v=1.
    """
    import cv2

    if len(values) < 2:
        return

    x, y = org
    w, h = size
    panel(frame, (x, y), (x + w, y + h), PANEL, alpha=0.5, radius=4)

    dot_r = 2
    pad = dot_r + 2
    top, plot_h = y + pad, h - pad * 2
    left, plot_w = x + pad, w - pad * 2
    if plot_h < 4 or plot_w < 4:
        return

    def py(v: float) -> int:
        return top + plot_h - int(np.clip(v, 0.0, 1.0) * plot_h)

    ty = py(threshold)
    # Dashed threshold line: a solid one reads as part of the trace.
    for dx in range(0, plot_w, 6):
        cv2.line(frame, (left + dx, ty), (left + min(dx + 3, plot_w), ty), DIM, 1, cv2.LINE_AA)

    vals = list(values)[-plot_w:]
    step = plot_w / max(1, len(vals) - 1)
    pts = [(int(left + i * step), py(v)) for i, v in enumerate(vals)]
    cv2.polylines(frame, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.circle(frame, pts[-1], dot_r, color, -1, cv2.LINE_AA)


def corner_box(frame: np.ndarray, box, color, thickness: int = 2, frac: float = 0.22) -> None:
    """Corner brackets instead of a full rectangle.

    Brackets leave the face visible, which matters when the point of the demo is
    for a viewer to compare a real face against a photo of it side by side.
    """
    import cv2

    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    n = max(8, int(min(box.width, box.height) * frac))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * n, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * n), color, thickness, cv2.LINE_AA)
