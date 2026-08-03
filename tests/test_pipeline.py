"""Tests for pipeline trust logic and overlay states.

The security-relevant behavior of this project is not "does the model score well"
— it's the decision logic wrapped around the score. These tests cover the parts
where a wrong answer means a printed photo gets treated as a person, using fakes
so they run without model weights or a camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.contracts import (
    Box,
    FaceObservation,
    Identity,
    LivenessResult,
)
from trainyourface.core.pipeline import FacePipeline, FrameResult

BOX = Box(x1=10, y1=10, x2=110, y2=110, score=0.99)


def live(p: float = 0.1, thr: float = 0.5) -> LivenessResult:
    return LivenessResult(spoof_probability=p, threshold=thr)


def known(name: str = "kaitlyn", sim: float = 0.8) -> Identity:
    return Identity(name=name, similarity=sim, threshold=0.36)


class TestTrustRequiresBoth:
    """`is_trustworthy` is the project thesis; it must fail closed."""

    def test_known_and_live_is_trusted(self):
        f = FaceObservation(box=BOX, identity=known(), liveness=live(0.05))
        assert f.is_trustworthy

    def test_known_but_spoofed_is_not_trusted(self):
        """The core case: a printed photo of an enrolled user matches perfectly."""
        f = FaceObservation(box=BOX, identity=known(sim=0.99), liveness=live(0.95))
        assert not f.is_trustworthy

    def test_live_but_unknown_is_not_trusted(self):
        f = FaceObservation(box=BOX, identity=Identity(name=None), liveness=live(0.02))
        assert not f.is_trustworthy

    def test_missing_liveness_is_not_trusted(self):
        """No liveness model must mean untrusted, not trusted-by-default.

        Defaulting to trusted here would make recognition-only mode silently
        claim a guarantee it can't provide.
        """
        f = FaceObservation(box=BOX, identity=known(), liveness=None)
        assert not f.is_trustworthy

    def test_nothing_known_is_not_trusted(self):
        assert not FaceObservation(box=BOX).is_trustworthy


class TestOverlayStates:
    def test_states_map_correctly(self):
        from trainyourface.cli.overlay import face_state

        cases = [
            (FaceObservation(box=BOX, identity=known(), liveness=live(0.01)), "TRUSTED"),
            (FaceObservation(box=BOX, identity=known(), liveness=live(0.99)), "SPOOF"),
            (FaceObservation(box=BOX, identity=Identity(), liveness=live(0.01)), "UNKNOWN"),
            (FaceObservation(box=BOX, identity=known(), liveness=None), "UNVERIFIED"),
            (FaceObservation(box=BOX), "UNVERIFIED"),
        ]
        for face, expected in cases:
            assert face_state(face)[0] == expected

    def test_spoof_outranks_a_match(self):
        """A matched identity must never turn a rejected face green."""
        from trainyourface.cli.overlay import GREEN, face_state

        face = FaceObservation(box=BOX, identity=known(sim=0.99), liveness=live(0.99))
        state, color = face_state(face)
        assert state == "SPOOF"
        assert color != GREEN

    def test_draw_face_does_not_crash_on_edge_boxes(self):
        """Boxes running off-frame are routine detector output."""
        from trainyourface.cli.overlay import draw_face

        pytest.importorskip("cv2")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for box in (
            Box(x1=-40, y1=-40, x2=30, y2=30, score=0.9),
            Box(x1=140, y1=100, x2=400, y2=300, score=0.9),
        ):
            draw_face(frame, FaceObservation(box=box, liveness=live()))

    def test_hud_renders_without_faces(self):
        from trainyourface.cli.overlay import draw_hud

        pytest.importorskip("cv2")
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        draw_hud(frame, FrameResult(faces=[], detect_ms=5.0), warn="no model")


# ---- fakes ---------------------------------------------------------------


class FakeDetector:
    spec = type("S", (), {"name": "fake"})()

    def __init__(self, boxes=None, kps=None):
        self._boxes = boxes if boxes is not None else [BOX]
        self._kps = kps if kps is not None else [np.zeros((5, 2), np.float32)]

    @property
    def active_provider(self):
        return "FakeProvider"

    def detect(self, image):
        return list(self._boxes), list(self._kps)


class FakeEmbedder:
    spec = type("S", (), {"name": "fake-embed"})()

    def __init__(self):
        self.calls = 0

    @property
    def active_provider(self):
        return "FakeProvider"

    def embed(self, crops):
        self.calls += 1
        v = np.zeros((len(crops), 512), dtype=np.float32)
        v[:, 0] = 1.0
        return v


class FakeLiveness:
    """Returns a fixed spoof probability for every face."""

    def __init__(self, prob: float, threshold: float = 0.5):
        self.prob = prob
        self.threshold = threshold
        self.path = type("P", (), {"name": "fake.onnx"})()

    @property
    def active_provider(self):
        return "FakeProvider"

    def check(self, frame, boxes):
        return [
            LivenessResult(spoof_probability=self.prob, threshold=self.threshold) for _ in boxes
        ]


@pytest.fixture
def frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


class TestPipelineGating:
    def test_spoof_short_circuits_recognition(self, frame):
        """A rejected face must not be embedded.

        Two reasons, both tested by the same assertion: it saves the expensive
        half of the pipeline, and it prevents the UI reporting 'SPOOF — matched
        Kaitlyn', which tells an attacker their spoof found the right target.
        """
        embedder = FakeEmbedder()
        pipe = FacePipeline(detector=FakeDetector(), embedder=embedder, liveness=FakeLiveness(0.99))
        result = pipe.process(frame)

        assert embedder.calls == 0
        assert result.faces[0].embedding is None
        assert result.faces[0].identity is None
        assert not result.faces[0].is_trustworthy

    def test_live_face_is_recognized(self, frame):
        embedder = FakeEmbedder()
        pipe = FacePipeline(detector=FakeDetector(), embedder=embedder, liveness=FakeLiveness(0.01))
        result = pipe.process(frame)
        assert embedder.calls == 1
        assert result.faces[0].embedding is not None

    def test_identify_spoofs_opt_in_still_reports_spoof(self, frame):
        """Eval mode embeds spoofs, but must not mark them trustworthy."""
        embedder = FakeEmbedder()
        pipe = FacePipeline(
            detector=FakeDetector(),
            embedder=embedder,
            liveness=FakeLiveness(0.99),
            identify_spoofs=True,
        )
        result = pipe.process(frame)
        assert embedder.calls == 1
        assert result.faces[0].embedding is not None
        assert not result.faces[0].is_trustworthy

    def test_no_liveness_model_recognizes_but_never_trusts(self, frame):
        embedder = FakeEmbedder()
        pipe = FacePipeline(detector=FakeDetector(), embedder=embedder, liveness=None)
        result = pipe.process(frame)
        assert embedder.calls == 1
        assert result.faces[0].liveness is None
        assert not result.faces[0].is_trustworthy

    def test_no_faces_skips_every_later_stage(self, frame):
        embedder = FakeEmbedder()
        pipe = FacePipeline(
            detector=FakeDetector(boxes=[], kps=[]),
            embedder=embedder,
            liveness=FakeLiveness(0.01),
        )
        result = pipe.process(frame)
        assert result.faces == []
        assert embedder.calls == 0

    def test_mixed_frame_gates_per_face(self, frame):
        """Two faces, one live one spoofed, must be handled independently."""

        class Mixed:
            path = type("P", (), {"name": "m.onnx"})()
            threshold = 0.5

            @property
            def active_provider(self):
                return "FakeProvider"

            def check(self, frame, boxes):
                return [
                    LivenessResult(spoof_probability=p, threshold=0.5)
                    for p in (0.01, 0.99)[: len(boxes)]
                ]

        boxes = [BOX, Box(x1=150, y1=20, x2=250, y2=120, score=0.9)]
        kps = [np.zeros((5, 2), np.float32)] * 2
        embedder = FakeEmbedder()
        pipe = FacePipeline(detector=FakeDetector(boxes, kps), embedder=embedder, liveness=Mixed())
        result = pipe.process(frame)

        assert len(result.faces) == 2
        assert result.faces[0].embedding is not None  # live
        assert result.faces[1].embedding is None  # spoof, skipped

    def test_largest_face_is_chosen_for_enrollment(self, frame):
        small = Box(x1=0, y1=0, x2=20, y2=20, score=0.99)
        big = Box(x1=50, y1=50, x2=200, y2=200, score=0.60)
        kps = [np.zeros((5, 2), np.float32)] * 2
        pipe = FacePipeline(detector=FakeDetector([small, big], kps))
        # Biggest, not highest-confidence — a background face shouldn't win.
        assert pipe.largest_face(frame).box == big

    def test_timings_are_populated(self, frame):
        pipe = FacePipeline(
            detector=FakeDetector(), embedder=FakeEmbedder(), liveness=FakeLiveness(0.01)
        )
        r = pipe.process(frame)
        assert r.detect_ms >= 0 and r.total_ms >= r.detect_ms
        assert r.fps > 0

    def test_describe_flags_missing_liveness(self):
        pipe = FacePipeline(detector=FakeDetector(), embedder=FakeEmbedder(), liveness=None)
        assert any("spoofs will NOT be detected" in line for line in pipe.describe())


class TestSoftmaxStability:
    def test_large_logits_do_not_overflow(self):
        """INT8 models can emit logits large enough to overflow a naive exp()."""
        from trainyourface.liveness.predict import _softmax_last

        out = _softmax_last(np.array([[1000.0, 999.0]], dtype=np.float32))
        assert np.all(np.isfinite(out))
        assert out.sum() == pytest.approx(1.0)

    def test_sums_to_one(self):
        from trainyourface.liveness.predict import _softmax_last

        rng = np.random.default_rng(0)
        out = _softmax_last(rng.normal(0, 50, (7, 2)))
        assert np.allclose(out.sum(axis=1), 1.0)
