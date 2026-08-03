"""Tests for the overlay rendering layer.

Drawing code is easy to leave untested because "it looks fine" is the usual
acceptance criterion, and these tests can't judge whether the UI looks good. What
they can do is catch the two failure modes that actually bite:

1. A crash on geometry the detector routinely produces — boxes off the top of the
   frame, boxes wider than the frame, degenerate boxes. The viewer runs this code
   once per face per frame, so an unhandled edge case is a hard crash mid-demo.
2. Drawing outside the region a caller sized for it. Two real bugs found this way
   are pinned below.

`draw_*` returning without raising is a weak assertion, so where a bug had a
measurable signature the test measures pixels instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.contracts import Box, FaceObservation, Identity, LivenessResult
from trainyourface.core.pipeline import FrameResult

pytest.importorskip("cv2")


def box(x1=40, y1=40, x2=160, y2=180, score=0.99) -> Box:
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def live(p: float = 0.1, thr: float = 0.5) -> LivenessResult:
    return LivenessResult(spoof_probability=p, threshold=thr)


@pytest.fixture
def frame() -> np.ndarray:
    # Mid-grey rather than black: it makes both dark shadows and light text
    # measurable against the background.
    return np.full((320, 480, 3), 60, np.uint8)


class TestTextRendering:
    """Regression tests for the shadow-overhang bug.

    The shadow was drawn at `weight + 1`. OpenCV renders a heavier stroke WIDER
    (68px vs 64px for one test string), so the shadow overhung the glyphs it sat
    behind and appeared as a dark duplicate letter trailing every string. It also
    overflowed panels, because `text_size` reports the thickness-1 width and every
    panel is sized from that.
    """

    def test_shadow_does_not_extend_past_reported_width(self, frame):
        from trainyourface.cli.theme import text, text_size

        s = "confirmed live"
        scale = 0.38
        w, _ = text_size(s, scale)
        text(frame, s, (10, 40), (255, 255, 255), scale=scale)

        # Nothing may be drawn at or beyond the advertised right edge.
        beyond = frame[:, 10 + w + 1 :]
        assert np.all(beyond == 60), "text or its shadow spilled past text_size()"

    def test_shadow_is_actually_drawn(self, frame):
        """The fix must not have silently removed the shadow.

        Without this, dropping the shadow entirely would pass the test above.
        """
        from trainyourface.cli.theme import text

        with_shadow = frame.copy()
        text(with_shadow, "abc", (10, 40), (255, 255, 255), scale=0.5, shadow=True)
        without = frame.copy()
        text(without, "abc", (10, 40), (255, 255, 255), scale=0.5, shadow=False)

        # Dark pixels exist only in the shadowed version.
        assert (with_shadow.min(axis=2) < 30).sum() > 0
        assert (with_shadow != without).any()

    def test_text_size_matches_what_gets_drawn(self, frame):
        from trainyourface.cli.theme import text, text_size

        s = "UNVERIFIED"
        w, h = text_size(s, 0.42)
        text(frame, s, (20, 100), (255, 255, 255), scale=0.42)
        ink = np.argwhere(frame.max(axis=2) > 100)
        assert ink.size > 0
        assert ink[:, 1].max() <= 20 + w
        assert ink[:, 0].min() >= 100 - h


class TestPanel:
    def test_panel_clipped_to_frame(self, frame):
        """Callers position panels from face boxes, which run off-frame."""
        from trainyourface.cli.theme import panel

        for tl, br in (((-80, -40), (60, 40)), ((400, 280), (900, 700)), ((0, 0), (480, 320))):
            panel(frame, tl, br)

    def test_degenerate_panel_is_a_noop(self, frame):
        from trainyourface.cli.theme import panel

        before = frame.copy()
        panel(frame, (100, 100), (100, 100))
        panel(frame, (100, 100), (101, 101))
        assert np.array_equal(frame, before)

    def test_panel_is_translucent_not_opaque(self, frame):
        """A fully opaque HUD hides part of the scene it's describing."""
        from trainyourface.cli.theme import panel

        panel(frame, (20, 20), (120, 80), (0, 0, 0), alpha=0.6)
        # Blended, so the region is darker than the background but not black.
        region = frame[40:60, 40:100]
        assert region.max() < 60
        assert region.min() > 0


class TestMeter:
    def test_threshold_tick_is_drawn(self, frame):
        from trainyourface.cli.theme import meter

        meter(frame, (100, 100), 200, 0.2, 0.5, (0, 255, 0))
        # The tick is white and sits above the bar, where nothing else draws.
        above = frame[95:99, 100:300]
        assert (above.min(axis=2) > 200).sum() > 0, "no tick above the bar"

    def test_negative_threshold_suppresses_the_tick(self, frame):
        """Progress bars reuse `meter` and must not show a decision mark.

        Clipping a negative threshold to 0.0 would put a tick at the bar's left
        edge, which is the bug this pins.
        """
        from trainyourface.cli.theme import meter

        meter(frame, (100, 100), 200, 0.2, -1.0, (0, 255, 0))
        band = frame[95:99, 90:310]
        assert (band.min(axis=2) > 200).sum() == 0, "a tick was drawn anyway"
        assert np.all(band == 60), "something was drawn outside the bar"


class TestSparkline:
    """Regression tests for the trace-on-the-border bug.

    Mapping 0..1 onto the full panel height put values near 0 and 1 exactly on the
    panel edge. A confident liveness model sits at ~0.02 or ~0.98 nearly all the
    time, so the normal case was the unreadable one.
    """

    @pytest.mark.parametrize("value", [0.0, 0.01, 0.5, 0.99, 1.0])
    def test_trace_stays_inside_the_panel(self, frame, value):
        from trainyourface.cli.theme import sparkline

        org, size = (100, 100), (92, 34)
        sparkline(frame, org, size, [value] * 20, 0.5, (0, 255, 0))

        drawn = np.argwhere(frame[:, :, 1] > 150)
        assert drawn.size > 0, "nothing was plotted"
        ys, xs = drawn[:, 0], drawn[:, 1]
        # Strictly inside: touching the border is the bug being tested for.
        assert ys.min() > org[1], f"value={value} drew on the top border"
        assert ys.max() < org[1] + size[1] - 1, f"value={value} drew on the bottom border"
        assert xs.min() >= org[0]
        assert xs.max() <= org[0] + size[0]

    def test_too_few_points_is_a_noop(self, frame):
        from trainyourface.cli.theme import sparkline

        before = frame.copy()
        sparkline(frame, (10, 10), (80, 30), [0.5], 0.5, (0, 255, 0))
        assert np.array_equal(frame, before)

    def test_tiny_panel_does_not_crash(self, frame):
        from trainyourface.cli.theme import sparkline

        sparkline(frame, (10, 10), (4, 4), [0.2, 0.8, 0.3], 0.5, (0, 255, 0))

    def test_more_points_than_pixels_is_downsampled(self, frame):
        """History is 120 frames; a narrow panel is fewer pixels than that."""
        from trainyourface.cli.theme import sparkline

        sparkline(frame, (10, 10), (40, 30), list(np.linspace(0, 1, 500)), 0.5, (0, 255, 0))


class TestScoreHistory:
    def test_series_accumulates_per_position(self):
        from trainyourface.cli.overlay import ScoreHistory

        h = ScoreHistory()
        b = box()
        for v in (0.1, 0.2, 0.3):
            h.push(b, v)
        assert list(h.get(b)) == [0.1, 0.2, 0.3]

    def test_distant_faces_get_separate_series(self):
        from trainyourface.cli.overlay import ScoreHistory

        h = ScoreHistory(cell=100)
        near, far = box(0, 0, 50, 50), box(400, 260, 470, 310)
        h.push(near, 0.1)
        h.push(far, 0.9)
        assert list(h.get(near)) == [0.1]
        assert list(h.get(far)) == [0.9]

    def test_history_is_bounded(self):
        """A long session must not grow the trace without limit."""
        from trainyourface.cli.overlay import HISTORY, ScoreHistory

        h = ScoreHistory()
        b = box()
        for i in range(HISTORY * 3):
            h.push(b, i / (HISTORY * 3))
        assert len(h.get(b)) == HISTORY

    def test_stale_series_are_evicted(self):
        """Otherwise a dict entry leaks per grid cell ever visited."""
        from trainyourface.cli.overlay import HISTORY, ScoreHistory

        h = ScoreHistory(cell=100)
        gone = box(0, 0, 50, 50)
        h.push(gone, 0.5)
        here = box(400, 260, 470, 310)
        for _ in range(HISTORY + 2):
            h.push(here, 0.5)
        assert list(h.get(gone)) == []

    def test_unknown_position_returns_empty(self):
        from trainyourface.cli.overlay import ScoreHistory

        assert list(ScoreHistory().get(box())) == []


class TestDrawFace:
    @pytest.mark.parametrize(
        "b",
        [
            box(-70, -70, 30, 30),  # off top-left
            box(440, 290, 700, 600),  # off bottom-right
            box(0, 0, 480, 320),  # fills the frame
            box(10, 2, 60, 30),  # too close to the top for a badge above it
            box(200, 300, 260, 318),  # at the bottom, no room for the meter below
            box(1, 1, 3, 3),  # degenerate
        ],
    )
    def test_edge_geometry_does_not_crash(self, frame, b):
        from trainyourface.cli.overlay import draw_face

        draw_face(frame, FaceObservation(box=b, liveness=live()))

    def test_identity_shown_only_when_trusted(self, frame):
        """A rejected spoof must not have its target confirmed on screen."""
        from trainyourface.cli.overlay import draw_face

        ident = Identity(name="kaitlyn", similarity=0.99, threshold=0.36)
        spoofed = frame.copy()
        draw_face(spoofed, FaceObservation(box=box(), identity=ident, liveness=live(0.99)))
        trusted = frame.copy()
        draw_face(trusted, FaceObservation(box=box(), identity=ident, liveness=live(0.01)))
        # The trusted render carries an extra identity badge, so it has strictly
        # more non-background pixels.
        assert (trusted != 60).sum() > (spoofed != 60).sum()

    def test_missing_liveness_draws_no_meter(self, frame):
        from trainyourface.cli.overlay import draw_face

        without = frame.copy()
        draw_face(without, FaceObservation(box=box()))
        withl = frame.copy()
        draw_face(withl, FaceObservation(box=box(), liveness=live()))
        assert (withl != 60).sum() > (without != 60).sum()

    def test_trace_can_be_disabled(self, frame):
        from trainyourface.cli.overlay import ScoreHistory, draw_face

        h = ScoreHistory()
        b = box()
        for i in range(40):
            h.push(b, i / 40)
        face = FaceObservation(box=b, liveness=live())

        on = frame.copy()
        draw_face(on, face, history=h, show_trace=True)
        off = frame.copy()
        draw_face(off, face, history=h, show_trace=False)
        assert (on != off).any()


class TestDrawHud:
    def test_renders_with_every_option(self, frame):
        from trainyourface.cli.overlay import draw_hud

        faces = [
            FaceObservation(
                box=box(),
                identity=Identity(name="k", similarity=0.9, threshold=0.36),
                liveness=live(0.01),
            ),
            FaceObservation(box=box(200, 40, 300, 180), liveness=live(0.99)),
        ]
        draw_hud(
            frame,
            FrameResult(faces=faces, detect_ms=18.0, liveness_ms=0.4, recognize_ms=1.4),
            warn="NO LIVENESS MODEL - a photo would pass",
            keys="q quit  s shot",
            subtitle="trainyourface",
            legend=True,
            paused=True,
            extra=["enrolling kaitlyn"],
        )

    def test_renders_with_no_faces_and_no_timings(self, frame):
        from trainyourface.cli.overlay import draw_hud

        draw_hud(frame, FrameResult(faces=[]))

    def test_renders_on_a_tiny_frame(self):
        """A low-resolution camera must not make the HUD raise."""
        from trainyourface.cli.overlay import draw_hud

        draw_hud(
            np.zeros((96, 128, 3), np.uint8),
            FrameResult(faces=[], detect_ms=5.0),
            warn="a warning long enough to exceed this frame's width by a lot",
            keys="q quit  s shot  SPACE pause  l legend",
            legend=True,
        )

    def test_every_state_has_a_hint(self):
        from trainyourface.cli.overlay import face_state, state_hint

        states = {
            face_state(f)[0]
            for f in (
                FaceObservation(
                    box=box(),
                    identity=Identity(name="k", similarity=0.9, threshold=0.36),
                    liveness=live(0.01),
                ),
                FaceObservation(box=box(), liveness=live(0.99)),
                FaceObservation(box=box(), liveness=live(0.01)),
                FaceObservation(box=box()),
            )
        }
        assert states == {"TRUSTED", "SPOOF", "UNKNOWN", "UNVERIFIED"}
        for s in states:
            assert state_hint(s), f"{s} has no explanation for the legend"


class TestCaptureHud:
    def test_states_render(self, frame):
        from trainyourface.cli.overlay import draw_capture_hud

        common = dict(subject="k", session="s1", target=120)
        for kwargs in (
            dict(label="bona_fide", captured=0, has_face=False, recording=False),
            dict(label="replay", captured=60, has_face=True, recording=True, fps=41.0),
            dict(label="print", captured=120, has_face=True, recording=False, reason="2 faces"),
        ):
            draw_capture_hud(frame.copy(), **common, **kwargs)

    def test_zero_target_does_not_divide_by_zero(self, frame):
        from trainyourface.cli.overlay import draw_capture_hud

        draw_capture_hud(
            frame,
            label="print",
            subject="k",
            session="s1",
            captured=0,
            target=0,
            has_face=True,
        )

    def test_no_face_is_reported_even_without_an_explicit_reason(self, frame):
        """has_face=False must produce a visible blocker message."""
        from trainyourface.cli.overlay import draw_capture_hud

        blocked = draw_capture_hud(
            frame.copy(),
            label="print",
            subject="k",
            session="s1",
            captured=0,
            target=10,
            has_face=False,
        )
        ok = draw_capture_hud(
            frame.copy(),
            label="print",
            subject="k",
            session="s1",
            captured=0,
            target=10,
            has_face=True,
            recording=True,
        )
        assert (blocked != ok).any()
