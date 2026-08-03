"""Validated data contracts shared across detection, recognition, and liveness.

Everything that crosses a module boundary is a pydantic model rather than a
tuple or a bare dict. Two reasons this matters here specifically:

1. The old version passed `(dlib_vector, dlib_rectangle)` tuples around and
   stored embeddings as untyped JSON lists. A shape or dtype mistake surfaced as
   a wrong recognition result, not an error. Validation moves those failures to
   the boundary where they're debuggable.
2. The PAD metrics are only meaningful if every prediction carries the attack
   type it was made against. Encoding that in the type system means the eval
   harness cannot silently average over a category it forgot about.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# The dlib ResNet emitted 128-D descriptors. ArcFace-family models — what we use
# now — emit 512-D. Pinned as a constant so a model swap that changes dimension
# fails loudly at load instead of quietly wrecking every distance computation.
EMBEDDING_DIM = 512


class AttackType(str, Enum):
    """Presentation attack categories, following the ISO/IEC 30107-3 vocabulary.

    APCER is reported *per attack type* because a detector can look excellent on
    average while being useless against one category. A model that catches 99% of
    printed photos and 10% of replayed video is not a 55%-good model — it's a
    model an attacker beats every time with a phone screen. Averaging hides that;
    keeping the category on every sample makes hiding it impossible.
    """

    # Not an attack — a real human in front of the camera. Called "bona fide" in
    # the standard, kept as the enum's zero-value for readable confusion matrices.
    BONA_FIDE = "bona_fide"
    # A photo printed on paper and held up to the camera.
    PRINT = "print"
    # A photo or video played back on a phone/tablet/monitor screen.
    REPLAY = "replay"
    # A printed photo with eye/mouth regions cut out, worn as a mask.
    CUTOUT = "cutout"
    # A wearable 3D mask (silicone, resin, paper craft).
    MASK_3D = "mask_3d"

    @property
    def is_attack(self) -> bool:
        return self is not AttackType.BONA_FIDE


class Box(BaseModel):
    """An axis-aligned face bounding box in pixel coordinates.

    Replaces `dlib.rectangle`, which was one of the two things (with the ResNet)
    that forced a dlib dependency. Coordinates are ints because they index pixels.
    """

    model_config = ConfigDict(frozen=True)

    x1: int
    y1: int
    x2: int
    y2: int
    score: float = Field(ge=0.0, le=1.0, description="Detector confidence.")

    @field_validator("x2")
    @classmethod
    def _x_ordered(cls, v: int, info) -> int:
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError(f"x2 ({v}) must exceed x1 ({info.data['x1']})")
        return v

    @field_validator("y2")
    @classmethod
    def _y_ordered(cls, v: int, info) -> int:
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError(f"y2 ({v}) must exceed y1 ({info.data['y1']})")
        return v

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def scaled(self, factor_x: float, factor_y: float) -> Box:
        """Rescale into another resolution's coordinate space.

        The old code detected on the full-resolution frame but drew on a resized
        one, so this rescaling was load-bearing for the overlay lining up. Kept
        as an explicit method rather than inline arithmetic at each call site.

        A small box scaled down rounds both edges to the same pixel, which would
        fail Box's own x2 > x1 validator — so the result is widened to a minimum of
        one pixel. Downscaling a detection is a legitimate operation and should not
        raise just because the source box was tiny.
        """
        x1, y1 = int(self.x1 * factor_x), int(self.y1 * factor_y)
        x2, y2 = int(self.x2 * factor_x), int(self.y2 * factor_y)
        return Box(
            x1=x1,
            y1=y1,
            x2=max(x2, x1 + 1),
            y2=max(y2, y1 + 1),
            score=self.score,
        )

    def clipped(self, width: int, height: int) -> Box:
        """Clamp to image bounds. Detectors routinely emit boxes running off-frame."""
        return Box(
            x1=max(0, min(self.x1, width - 2)),
            y1=max(0, min(self.y1, height - 2)),
            x2=max(1, min(self.x2, width - 1)),
            y2=max(1, min(self.y2, height - 1)),
            score=self.score,
        )


class LivenessResult(BaseModel):
    """Verdict from the PAD model for one face.

    `spoof_probability` is kept alongside the boolean because the boolean is
    threshold-dependent and the threshold is a deployment choice: an unlocking
    app wants a low BPCER (don't annoy the real user), a door lock wants a low
    APCER (don't admit an attacker). Returning the raw score lets callers pick
    their own operating point instead of inheriting ours.
    """

    model_config = ConfigDict(frozen=True)

    spoof_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    # Populated only when the model is multi-class; a binary real/spoof model
    # leaves this None rather than guessing a category it can't distinguish.
    predicted_attack_type: AttackType | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)

    @property
    def is_live(self) -> bool:
        return self.spoof_probability < self.threshold

    @property
    def label(self) -> str:
        return "REAL" if self.is_live else "SPOOF"


class Identity(BaseModel):
    """A recognition match, or an explicit non-match.

    The old code returned `None` from `find_face_match` for three distinct
    situations — empty database, no face close enough, and an internal miss — so
    callers could not tell "I don't know this person" from "I have no data at
    all". This makes the distinction explicit and carries the score that drove it.
    """

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    similarity: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity to the best enrolled embedding.",
    )
    threshold: float = 0.0
    # Distinguishes "database is empty" from "nobody matched".
    database_empty: bool = False

    @property
    def is_known(self) -> bool:
        return self.name is not None

    @property
    def display_name(self) -> str:
        return self.name if self.name else "UNKNOWN"


class FaceObservation(BaseModel):
    """One detected face in one frame, with everything we concluded about it.

    This is the single object the UI renders and the API returns. Recognition and
    liveness are both optional because the pipeline is staged: a caller running
    detection only, or liveness-gated recognition that short-circuited on a
    spoof, both produce valid partial observations.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    box: Box
    embedding: np.ndarray | None = None
    identity: Identity | None = None
    liveness: LivenessResult | None = None

    @field_validator("embedding")
    @classmethod
    def _check_embedding(cls, v: np.ndarray | None) -> np.ndarray | None:
        """Fail loudly on a wrong-shaped embedding.

        Without this, a model swap emitting 128-D instead of 512-D would produce
        plausible-looking but meaningless similarities rather than an error.
        """
        if v is None:
            return v
        if v.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {v.shape}")
        if v.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"embedding must be {EMBEDDING_DIM}-D, got {v.shape[0]}-D")
        return v.astype(np.float32, copy=False)

    @property
    def is_trustworthy(self) -> bool:
        """True only if this face is BOTH a known identity AND confirmed live.

        The project's whole thesis in one property. A recognized face is not a
        trusted face: the printed photo of an enrolled user matches the enrolled
        user perfectly. Recognition without liveness answers "does this look like
        Kaitlyn" when the question was "is Kaitlyn here".
        """
        if self.identity is None or not self.identity.is_known:
            return False
        # Absent a liveness model we deliberately return False rather than
        # defaulting to trusted — fail closed, not open.
        if self.liveness is None:
            return False
        return self.liveness.is_live
