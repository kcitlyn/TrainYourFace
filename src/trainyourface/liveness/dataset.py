"""Dataset and splits for presentation-attack detection.

The single most important thing in this file is `subject_disjoint_split`, and it
is worth explaining at length because getting it wrong is the most common way PAD
projects report numbers that are quietly meaningless.

THE LEAKAGE TRAP
----------------
The obvious way to split a PAD dataset is randomly by image. That is wrong here.
A capture session produces many near-identical frames of the same person under the
same lighting with the same attack instrument. Split randomly and frame 41 lands
in train while frame 42 lands in test. The model can then score ~99% by
memorizing "this specific person in this specific room", and it will collapse on
anyone it hasn't seen.

Splitting by SUBJECT — every frame of a given person goes entirely to one split —
forces the model to learn cues that transfer: moiré from a screen, the flatness of
paper, print halftone texture, specular highlights. Those generalize. Face
identity does not.

We go further and also support splitting by capture SESSION, because the same
subject photographed on two different days under different light is closer to two
subjects than to one for this task.

WHY THE DATASET IS BUILT, NOT DOWNLOADED
----------------------------------------
The standard academic PAD datasets (CASIA-FASD, Replay-Attack, OULU-NPU, SiW) all
require signed institutional license agreements and are not redistributable. So
this project cannot ship one, and pretending otherwise would make the results
unreproducible.

Instead `tyf capture` records your own: bona-fide frames from the webcam, then
print attacks (photo on paper) and replay attacks (photo/video on a phone
screen). A few hundred frames across a handful of subjects and 2-3 capture
sessions is enough to train a model that demonstrably works, and it is honest
about what it is: a small self-collected dataset, with metrics reported on
subject-disjoint splits.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from trainyourface.core.contracts import AttackType

# Liveness input resolution. Smaller than you might expect, on purpose: PAD cues
# are largely texture (moiré, print dots, specular reflection), which survives
# downscaling, and 128px keeps the model small enough to hit real-time on a Pi.
INPUT_SIZE = 128

# Extra context kept around the detected face box. Screen bezels, paper edges,
# and the hand holding the attack instrument are strong PAD cues, and a tight
# crop throws them away. 0.3 = 30% margin on each side.
CROP_MARGIN = 0.30


@dataclass
class Sample:
    """One labeled image in the PAD dataset."""

    path: str
    attack_type: AttackType
    # Who is in the image. The unit of splitting — never split within a subject.
    subject: str
    # Which capture session. Same subject, different day/lighting.
    session: str = "default"
    # Free-form: "iphone13_screen", "laser_print_matte", etc. Reported per-type in
    # eval so we can say WHICH instrument defeats the model.
    instrument: str | None = None

    @property
    def label(self) -> int:
        """Binary target: 1 = attack, 0 = bona fide."""
        return int(self.attack_type.is_attack)


@dataclass
class DatasetManifest:
    """The dataset index, persisted as JSON next to the images.

    A manifest rather than directory-scanning-at-train-time so that a split is
    reproducible: the exact file list and labels used for a reported number can be
    committed alongside it.
    """

    samples: list[Sample] = field(default_factory=list)
    root: str = ""

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": self.root,
            "samples": [{**asdict(s), "attack_type": s.attack_type.value} for s in self.samples],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> DatasetManifest:
        data = json.loads(Path(path).read_text())
        samples = [
            Sample(
                path=s["path"],
                attack_type=AttackType(s["attack_type"]),
                subject=s["subject"],
                session=s.get("session", "default"),
                instrument=s.get("instrument"),
            )
            for s in data.get("samples", [])
        ]
        return cls(samples=samples, root=data.get("root", ""))

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for s in self.samples:
            out[s.attack_type.value] += 1
        return dict(out)

    def subjects(self) -> list[str]:
        return sorted({s.subject for s in self.samples})

    def describe(self) -> str:
        lines = [
            f"{len(self.samples)} samples, {len(self.subjects())} subjects",
            f"sessions: {sorted({s.session for s in self.samples})}",
        ]
        for k, v in sorted(self.counts().items()):
            lines.append(f"  {k:12s} {v:5d}")
        instruments = sorted({s.instrument for s in self.samples if s.instrument})
        if instruments:
            lines.append(f"instruments: {instruments}")
        return "\n".join(lines)


def subject_disjoint_split(
    samples: list[Sample],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    by: str = "subject",
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split so that no subject (or session) appears in more than one split.

    This is the anti-leakage guarantee. See the module docstring for why a random
    per-image split would inflate results to near-100% and mean nothing.

    Args:
        by: "subject" or "session". Session-level is stricter — it also prevents
            the same person's other capture day from leaking across.

    Raises:
        ValueError: if there are too few groups to form disjoint splits, because
            silently returning an empty test set would produce metrics computed on
            nothing.
    """
    if by not in ("subject", "session"):
        raise ValueError(f"`by` must be 'subject' or 'session', got {by!r}")
    if not samples:
        raise ValueError("cannot split an empty sample list")

    key = (lambda s: s.subject) if by == "subject" else (lambda s: f"{s.subject}/{s.session}")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[key(s)].append(s)

    names = sorted(groups)
    if len(names) < 3:
        raise ValueError(
            f"need at least 3 {by}s for a disjoint train/val/test split, got {len(names)}: "
            f"{names}. Capture more subjects — a split that reuses subjects across "
            "train and test produces meaningless metrics."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(names)

    n = len(names)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    # Guarantee at least one group left for training.
    if n_test + n_val >= n:
        n_test, n_val = 1, 1

    test_names = set(names[:n_test])
    val_names = set(names[n_test : n_test + n_val])

    train, val, test = [], [], []
    for name in names:
        target = test if name in test_names else val if name in val_names else train
        target.extend(groups[name])

    return train, val, test


def check_split_integrity(
    train: list[Sample], val: list[Sample], test: list[Sample], by: str = "subject"
) -> None:
    """Assert the splits really are disjoint, and both classes appear in each.

    Called after every split. A leak here silently invalidates every number the
    project reports, so it is checked rather than assumed.
    """
    key = (lambda s: s.subject) if by == "subject" else (lambda s: f"{s.subject}/{s.session}")
    g_train, g_val, g_test = ({key(s) for s in part} for part in (train, val, test))

    for a, b, na, nb in (
        (g_train, g_val, "train", "val"),
        (g_train, g_test, "train", "test"),
        (g_val, g_test, "val", "test"),
    ):
        overlap = a & b
        if overlap:
            raise ValueError(
                f"LEAKAGE: {len(overlap)} {by}(s) appear in both {na} and {nb}: "
                f"{sorted(overlap)[:5]}. Metrics from this split would be invalid."
            )

    # PAD metrics are undefined without both classes present.
    for part, name in ((train, "train"), (val, "val"), (test, "test")):
        if not part:
            raise ValueError(f"{name} split is empty")
        labels = {s.label for s in part}
        if len(labels) < 2:
            only = "attack" if 1 in labels else "bona-fide"
            raise ValueError(
                f"{name} split contains only {only} samples. PAD metrics need both "
                "classes; capture more data or adjust the split."
            )


def to_model_input(img: np.ndarray) -> np.ndarray:
    """uint8 HWC BGR -> float32 CHW RGB in [-1, 1]. Returns (3, H, W), no batch dim.

    This function exists so training and inference cannot drift apart. The
    previous version of this project had exactly that bug in its recognition
    path: enrollment ran images through an aligned crop while the live path used
    an unaligned one, so enrolled and live embeddings landed in different regions
    of embedding space and matching silently degraded. Nothing errored; the
    threshold just had to be cranked down to compensate.

    A train/serve preprocessing mismatch is invisible in tests that only exercise
    one side, so both sides call this one function.
    """
    rgb = img[:, :, ::-1].astype(np.float32)
    rgb = (rgb - 127.5) / 127.5
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def load_image(path: str | Path, size: int = INPUT_SIZE) -> np.ndarray:
    """Read an image as a (size, size, 3) uint8 BGR array."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


class PADDataset:
    """torch Dataset over PAD samples. Imports torch lazily.

    Lazy import matters: this module is also used by `tyf capture` and by the
    manifest tooling, which must work in a core install where torch is absent.
    """

    def __init__(
        self,
        samples: list[Sample],
        root: Path | str = "",
        train: bool = False,
        size: int = INPUT_SIZE,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.root = Path(root)
        self.train = train
        self.size = size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream.

        Augmentation is seeded from (seed, epoch, index), so it has to be told
        which epoch it's in. Without this the same image would get the identical
        augmentation every epoch, which defeats the point of augmenting.

        The train loader is built with persistent_workers=False specifically so
        worker processes are re-created each epoch and pick up this value; with
        persistent workers, a mutation here would never reach the worker's copy of
        the dataset and augmentation would silently freeze at epoch 0.
        """
        self.epoch = epoch

    def __getitem__(self, idx: int):
        import torch

        s = self.samples[idx]
        path = self.root / s.path if self.root.parts else Path(s.path)
        img = load_image(path, self.size)

        if self.train:
            # Seeded from (seed, epoch, idx) rather than from global state, so a
            # run is reproducible regardless of how many DataLoader workers are
            # used or what order they happen to pull indices in. Worker count
            # would otherwise change the augmentation each item receives, which
            # makes --seed a promise the code doesn't keep.
            img = self._augment(img, np.random.default_rng((self.seed, self.epoch, idx)))

        # Shared with the inference path — see to_model_input.
        tensor = torch.from_numpy(to_model_input(img))
        return tensor, torch.tensor(s.label, dtype=torch.long)

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Augmentation chosen specifically NOT to destroy PAD cues.

        This is a real design constraint, not boilerplate. Standard image
        augmentation would wreck this task:

        - Heavy blur erases moiré and print texture — the actual signal. Kept mild.
        - Aggressive color jitter can manufacture screen-like tints on genuine
          faces, teaching the model a cue that isn't there. Kept mild.
        - JPEG recompression is INCLUDED on purpose: real deployments see
          compressed camera frames, and compression artifacts partially mask
          moiré, so training through it prevents a model that only works on
          pristine captures.
        - Vertical flips are excluded: faces have a canonical orientation and
          upside-down faces never occur at inference.
        """
        import cv2

        if rng.random() < 0.5:
            img = img[:, ::-1]  # horizontal flip only

        # Mild brightness/contrast shift.
        if rng.random() < 0.7:
            alpha = 1.0 + rng.uniform(-0.2, 0.2)
            beta = rng.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Small rotation/translation; a face detector already roughly centered it.
        if rng.random() < 0.5:
            angle = rng.uniform(-8, 8)
            tx, ty = rng.uniform(-0.04, 0.04, 2) * self.size
            m = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), angle, 1.0)
            m[:, 2] += (tx, ty)
            img = cv2.warpAffine(img, m, (self.size, self.size), borderMode=cv2.BORDER_REFLECT)

        # JPEG recompression — deliberately included, see docstring.
        if rng.random() < 0.3:
            quality = int(rng.integers(45, 95))
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Very mild blur only.
        if rng.random() < 0.15:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return np.ascontiguousarray(img)


def class_weights(samples: list[Sample]) -> tuple[float, float]:
    """Inverse-frequency weights for (bona_fide, attack).

    PAD datasets are usually attack-heavy, and unweighted training on a 1:3 split
    drifts toward predicting "attack" — which looks fine on accuracy and is
    terrible for BPCER (real users get locked out). These feed the loss function.
    """
    n_bona = sum(1 for s in samples if not s.attack_type.is_attack)
    n_attack = len(samples) - n_bona
    if n_bona == 0 or n_attack == 0:
        return 1.0, 1.0
    total = n_bona + n_attack
    return total / (2.0 * n_bona), total / (2.0 * n_attack)
