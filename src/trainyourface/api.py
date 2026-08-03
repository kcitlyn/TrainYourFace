"""The public Python API: three classes over the pieces the CLI already uses.

WHY THIS EXISTS
---------------
Until now `import trainyourface` gave you a version string. Everything real lived
under `trainyourface.core.*` and had to be wired together by hand — which means
this was a tool you could run, not a library you could build on.

The shape here is a direct response to what people actually complain about in the
incumbent (DeepFace). Two of its most-requested anti-spoofing issues were closed
without a fix:

  - liveness raised an exception on a spoof instead of returning a flag, so every
    call site needed a try/except to handle the normal case
  - the model reloaded on every call, which is untenable in a server or a video
    loop

Both are design decisions, not bugs, and this API takes the other branch on each:
a spoof is a RETURN VALUE, and models load once in __init__ and stay loaded.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No `verify()` that returns a bare bool. The whole argument of this project is that
"is this face real" and "is this the right person" are different questions, and
collapsing them into one boolean is how the original version of this project ended
up greeting a printed photo by name. `TrustedFace` keeps them separate and makes
the combined judgment explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["LivenessDetector", "FaceID", "TrustedFace"]


@dataclass(frozen=True)
class TrustedFace:
    """One face, with the two questions answered separately.

    `is_trusted` requires BOTH — recognized and verified live. Unknown liveness
    counts as untrusted rather than as a pass, because the failure mode of the
    opposite choice is a photo unlocking a door.
    """

    box: tuple[int, int, int, int]
    name: str | None
    similarity: float | None
    is_live: bool | None
    spoof_score: float | None

    @property
    def is_trusted(self) -> bool:
        """Recognized AND live. The only state that means anything."""
        return self.name is not None and self.is_live is True

    @property
    def status(self) -> str:
        """One of TRUSTED / SPOOF / UNKNOWN / UNVERIFIED, matching `tyf watch`.

        Kept identical to the CLI's vocabulary on purpose: a library and its own
        tool disagreeing about what to call a state is a documentation problem
        waiting to happen.
        """
        if self.is_live is False:
            return "SPOOF"
        if self.is_live is None:
            return "UNVERIFIED"
        return "TRUSTED" if self.name is not None else "UNKNOWN"


class LivenessDetector:
    """Is this a real face, or a photo/screen? Loads once, reuses the session.

    >>> det = LivenessDetector()
    >>> result = det.check(frame)          # frame: HWC BGR uint8, e.g. from cv2
    >>> result.is_live, result.spoof_probability
    (True, 0.02)

    Raises at construction if no model is available, rather than degrading to a
    detector that passes everything. A liveness check that silently always returns
    "live" is worse than no check, because it looks like protection.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        from trainyourface.core.detect import FaceDetector
        from trainyourface.liveness.predict import (
            LivenessModel,
            LivenessUnavailable,
            default_model_path,
        )

        path = Path(model_path) if model_path else default_model_path()
        try:
            self.model = LivenessModel(path, threshold=threshold, prefer_gpu=prefer_gpu)
        except LivenessUnavailable as exc:
            # Re-raised as FileNotFoundError so callers can catch a stdlib type
            # rather than importing an internal one, but the message is kept: it
            # already names the fix.
            raise FileNotFoundError(str(exc)) from exc
        self._detector = FaceDetector(prefer_gpu=prefer_gpu)

    @property
    def threshold(self) -> float:
        return self.model.threshold

    def check(self, frame: np.ndarray, detect: bool = True):
        """Score the largest face in a frame.

        Args:
            detect: run face detection first (the normal case). Pass False when
                `frame` is already a cropped face, e.g. from your own detector.

        Returns a LivenessResult with `.is_live`, `.spoof_probability`, `.label`.
        Returns None when detection finds no face — an empty frame is not a spoof,
        and returning a verdict for a face that isn't there would be a fabrication.
        """
        from trainyourface.core.contracts import LivenessResult

        if not detect:
            score = float(self.model.spoof_probability([frame])[0])
            return LivenessResult(
                spoof_probability=score,
                threshold=self.model.threshold,
            )
        boxes, _ = self._detector.detect(frame)
        if not boxes:
            return None
        largest = max(boxes, key=lambda b: b.area)
        return self.model.check(frame, [largest])[0]

    def check_all(self, frame: np.ndarray) -> list:
        """Score every detected face. Empty list when there are none."""
        boxes, _ = self._detector.detect(frame)
        return list(self.model.check(frame, boxes)) if boxes else []


class FaceID:
    """Recognition gated behind liveness, which is the point of the project.

    >>> fid = FaceID()
    >>> fid.enroll("kaitlyn", frame)       # refuses a photo
    >>> for face in fid.identify(frame):
    ...     print(face.status, face.name)
    TRUSTED kaitlyn

    A spoofed face is never named. Not merely reported-and-ignored — recognition
    does not run on it, so nothing downstream can leak "SPOOF, matched kaitlyn"
    and tell an attacker they found the right target.
    """

    def __init__(
        self,
        store_path: Path | str | None = None,
        liveness: LivenessDetector | Path | str | None = None,
        match_threshold: float | None = None,
        require_liveness: bool = True,
    ) -> None:
        from trainyourface.core.detect import FaceDetector
        from trainyourface.core.embed import FaceEmbedder
        from trainyourface.core.pipeline import FacePipeline
        from trainyourface.core.store import EnrollmentStore, default_store_path

        self.require_liveness = require_liveness

        model = None
        if isinstance(liveness, LivenessDetector):
            model = liveness.model
        elif liveness is not None:
            model = LivenessDetector(liveness).model
        else:
            try:
                model = LivenessDetector().model
            except FileNotFoundError:
                # Recognition-only is a legitimate configuration, but it must be
                # asked for. Defaulting to it would mean `FaceID()` silently
                # accepts photos on a machine with no model installed.
                if require_liveness:
                    raise FileNotFoundError(
                        "no liveness model found, so faces could not be verified as "
                        "real. Train one with `tyf train`, or pass "
                        "require_liveness=False to accept that a photo will pass."
                    ) from None

        self.store = EnrollmentStore(Path(store_path) if store_path else default_store_path())
        self.pipeline = FacePipeline(
            detector=FaceDetector(),
            embedder=FaceEmbedder(),
            liveness=model,
            store=self.store,
            match_threshold=match_threshold,
        )

    def identify(self, frame: np.ndarray) -> list[TrustedFace]:
        """Every face in the frame, with liveness and identity resolved."""
        result = self.pipeline.process(frame)
        out = []
        for f in result.faces:
            live = f.liveness
            ident = f.identity
            out.append(
                TrustedFace(
                    box=(f.box.x1, f.box.y1, f.box.x2, f.box.y2),
                    name=getattr(ident, "name", None),
                    similarity=getattr(ident, "similarity", None),
                    is_live=None if live is None else live.is_live,
                    spoof_score=getattr(live, "spoof_probability", None),
                )
            )
        return out

    def enroll(self, name: str, frame: np.ndarray) -> TrustedFace:
        """Register a face. Raises rather than enrolling something unverified.

        Enrollment is the one operation where a bad input poisons everything after
        it: a photo enrolled as a person means that person's identity is now
        matchable by that photo forever, and a soft or blurry embedding degrades
        every comparison made against it. So the checks here raise instead of
        warning.
        """
        faces = self.identify(frame)
        if not faces:
            raise ValueError("no face detected in the frame")
        if len(faces) > 1:
            raise ValueError(
                f"{len(faces)} faces detected. Enrollment needs exactly one, "
                "otherwise it is ambiguous which person is being registered."
            )
        face = faces[0]
        if self.require_liveness and face.is_live is not True:
            raise ValueError(
                f"refusing to enroll: liveness says {face.status}. Enrolling from "
                "what might be a photo would make that photo a valid credential."
            )

        result = self.pipeline.process(frame)
        embedding = result.faces[0].embedding
        if embedding is None:
            raise ValueError("face detected but not embedded; cannot enroll")
        self.store.add(name, embedding)
        self.store.save()
        return face

    def forget(self, name: str) -> int:
        """Delete an identity. Returns how many embeddings were removed."""
        n = self.store.remove(name)
        self.store.save()
        return n

    @property
    def names(self) -> list[str]:
        """Enrolled identity names."""
        return list(self.store.identities())
