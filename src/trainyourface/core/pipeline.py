"""The end-to-end face pipeline: detect -> liveness -> align -> embed -> identify.

ORDERING: LIVENESS BEFORE RECOGNITION
-------------------------------------
Liveness runs first and, by default, short-circuits recognition on a spoof. Two
reasons, one of them a security property:

1. Cost. Recognition (align + 512-D embedding + matmul) is the expensive half. A
   rejected frame skips it, which is what makes the pipeline hold real-time on a
   Pi while both models are loaded.

2. Information leakage. If a spoof frame still gets identified, the UI can report
   "SPOOF — matched Kaitlyn (0.91)", which tells an attacker holding a printed
   photo that they have the right target and only need a better attack. Rejecting
   before identification means a failed attack learns nothing about who is
   enrolled. `identify_spoofs=True` is available for evaluation runs, where you
   specifically want to confirm that an attack WOULD have matched — that's the
   whole point of the project — but it is off by default.

The default is also the conservative one: recognition-only mode leaves
`liveness=None`, and `FaceObservation.is_trustworthy` treats unknown liveness as
untrusted. Failing closed matters more here than convenience, since the failure
mode is "printed photo unlocks the door".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from trainyourface.core.align import align_face, crop_box
from trainyourface.core.contracts import FaceObservation
from trainyourface.core.detect import FaceDetector
from trainyourface.core.embed import FaceEmbedder
from trainyourface.core.store import EnrollmentStore


@dataclass
class FrameResult:
    """Everything concluded about one frame, plus where the time went.

    Per-stage timings are separated because a single end-to-end number can't tell
    you whether a slow frame is the detector or the embedder, and that determines
    what you'd optimize.
    """

    faces: list[FaceObservation]
    detect_ms: float = 0.0
    liveness_ms: float = 0.0
    recognize_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.liveness_ms + self.recognize_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


class FacePipeline:
    """Composes the stages. Models are loaded once and reused across frames."""

    def __init__(
        self,
        *,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
        liveness=None,
        store: EnrollmentStore | None = None,
        match_threshold: float | None = None,
        identify_spoofs: bool = False,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.embedder = embedder
        self.liveness = liveness
        self.store = store
        self.match_threshold = match_threshold
        self.identify_spoofs = identify_spoofs

    def process(self, frame: np.ndarray) -> FrameResult:
        """Run the pipeline on one BGR frame."""
        t0 = time.perf_counter()
        boxes, keypoints = self.detector.detect(frame)
        detect_ms = (time.perf_counter() - t0) * 1000.0

        if not boxes:
            return FrameResult(faces=[], detect_ms=detect_ms)

        # ---- liveness ------------------------------------------------------
        t1 = time.perf_counter()
        liveness_results: list = [None] * len(boxes)
        if self.liveness is not None:
            scored = self.liveness.check(frame, boxes)
            # A backend returning a different number of verdicts than boxes is a
            # batching bug, and it must not become a wrong verdict. Everything
            # downstream indexes results by box position, so a short list silently
            # shifts one face's verdict onto another face — a spoof could be
            # attributed to the live person beside it. Fail loudly instead.
            if len(scored) != len(boxes):
                raise RuntimeError(
                    f"liveness backend returned {len(scored)} results for {len(boxes)} "
                    "faces; verdicts are matched to faces by position, so a mismatch "
                    "would assign the wrong verdict to a face"
                )
            liveness_results = list(scored)
        liveness_ms = (time.perf_counter() - t1) * 1000.0

        # ---- recognition ---------------------------------------------------
        # Only faces that passed liveness (or all of them, if there's no liveness
        # model / we're explicitly evaluating spoofs).
        t2 = time.perf_counter()
        to_identify = [
            i
            for i, live in enumerate(liveness_results)
            if self.identify_spoofs or live is None or live.is_live
        ]

        embeddings: dict[int, np.ndarray] = {}
        identities: dict[int, object] = {}

        if self.embedder is not None and to_identify:
            crops = []
            for i in to_identify:
                kps = keypoints[i] if i < len(keypoints) and keypoints[i] is not None else None
                # align_face needs 5 keypoints; without them fall back to a plain
                # box crop. Worse embeddings, but a detection isn't dropped just
                # because a keypoint-less detector variant is in use.
                crops.append(
                    align_face(frame, kps) if kps is not None else crop_box(frame, boxes[i])
                )

            matrix = self.embedder.embed(crops)
            for slot, i in enumerate(to_identify):
                embeddings[i] = matrix[slot]

            if self.store is not None and not self.store.is_empty:
                matches = self.store.identify_batch(matrix, threshold=self.match_threshold)
                for slot, i in enumerate(to_identify):
                    identities[i] = matches[slot]
        recognize_ms = (time.perf_counter() - t2) * 1000.0

        faces = [
            FaceObservation(
                box=boxes[i],
                embedding=embeddings.get(i),
                identity=identities.get(i),
                liveness=liveness_results[i],
            )
            for i in range(len(boxes))
        ]
        return FrameResult(faces, detect_ms, liveness_ms, recognize_ms)

    def largest_face(self, frame: np.ndarray) -> FaceObservation | None:
        """The biggest detected face, which is the enrollment subject.

        Enrollment needs exactly one face and "closest to the camera" is the
        reliable proxy — picking the highest-confidence box instead can select a
        background face the user didn't intend to enroll.
        """
        result = self.process(frame)
        if not result.faces:
            return None
        return max(result.faces, key=lambda f: f.box.area)

    def describe(self) -> list[str]:
        """Which models and providers are actually loaded, for `tyf info`."""
        lines = [f"detector    {self.detector.spec.name} [{self.detector.active_provider}]"]
        if self.embedder is not None:
            lines.append(f"embedder    {self.embedder.spec.name} [{self.embedder.active_provider}]")
        else:
            lines.append("embedder    (not loaded)")
        if self.liveness is not None:
            lines.append(
                f"liveness    {self.liveness.path.name} [{self.liveness.active_provider}] "
                f"threshold={self.liveness.threshold:.4f}"
            )
        else:
            lines.append("liveness    (not loaded) -- spoofs will NOT be detected")
        return lines
