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

WHERE THE DATA COMES FROM
-------------------------
Two routes, and they answer different questions.

`tyf import-celeba` converts CelebA-Spoof (625K images, 10,177 subjects, direct
download, no license application) into a manifest. This is the route to a real
benchmark number. Most of the classic PAD datasets — CASIA-FASD, Replay-Attack,
OULU-NPU, SiW — do require signed institutional agreements, which is why this
project long assumed a public one didn't exist; CelebA-Spoof is the exception.

`tyf capture` records your own from a webcam: bona-fide frames, then print
attacks (photo on paper) and replay attacks (photo/video on a screen).

The pair is worth more than either alone, because the honest question for a PAD
model is not "how well does it score on the data it was built from" — every
published model looks good there — but "does it survive a camera and a room it
has never seen". Train on CelebA-Spoof, test on your own captures, and the drop
between those two numbers is the result actually worth reporting. See
`tyf eval --cross` for that protocol.
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
    # Free-form capture conditions, e.g. {"illumination": "back", "environment":
    # "outdoor"}. Kept as an open dict rather than named fields because the
    # interesting axes differ per dataset, and eval stratifies by whatever keys
    # are present.
    #
    # This exists because a single aggregate APCER hides the failure that matters.
    # A model at 3% overall can be at 30% in backlit conditions, and backlit is
    # exactly where someone holds up a phone screen. Reporting per-condition is
    # what turns "it works" into "here is where it stops working".
    conditions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the path separator to forward slashes.

        A manifest is meant to be a portable, committable artifact — the file list
        behind a reported number. `tyf capture` built its paths with
        `str(Path("images") / label / name)`, which on Windows yields
        `images\\bona_fide\\s1_0.png`. Read back on macOS or Linux that is not a
        path at all: it is a SINGLE filename containing backslashes, so every image
        read fails and `_subject_from_path` can't find the identity directory.

        Normalizing here rather than at each writer means every construction site
        is covered, including future ones. Forward slashes are the right canonical
        form because Windows accepts them too, so the value stays usable on the
        platform that produced it.
        """
        if "\\" in self.path:
            self.path = self.path.replace("\\", "/")

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
        """Read a manifest, validating it rather than trusting it.

        A manifest is an on-disk file that gets hand-edited, generated by scripts,
        merged in git, and copied between machines, so "it parsed" is not the same
        as "it is usable". Every failure below used to surface as a raw
        JSONDecodeError, KeyError, or AttributeError — naming a dict key or a
        character offset instead of the file and the field.

        The `conditions` check earns its place: a non-dict value passed straight
        through here and then died in `fairness_audit` as
        `AttributeError: 'str' object has no attribute 'items'`, arbitrarily far
        from the manifest that caused it and with nothing pointing back to it.
        """
        path = Path(path)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}). A "
                "manifest is written by `tyf capture` or `tyf import-celeba`; if you "
                "edited it by hand, check for a trailing comma or a truncated write."
            ) from exc
        except OSError as exc:
            raise ValueError(f"could not read {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"{path} must contain a JSON object with 'samples' and 'root', got "
                f"{type(data).__name__}."
            )

        raw_samples = data.get("samples", [])
        if not isinstance(raw_samples, list):
            raise ValueError(f"{path}: 'samples' must be a list, got {type(raw_samples).__name__}")

        samples = []
        for i, s in enumerate(raw_samples):
            if not isinstance(s, dict):
                raise ValueError(f"{path}: sample {i} must be an object, got {type(s).__name__}")
            missing = [k for k in ("path", "attack_type", "subject") if k not in s]
            if missing:
                raise ValueError(f"{path}: sample {i} is missing required field(s) {missing}")
            try:
                attack_type = AttackType(s["attack_type"])
            except ValueError as exc:
                valid = [t.value for t in AttackType]
                raise ValueError(
                    f"{path}: sample {i} has attack_type {s['attack_type']!r}, which is "
                    f"not one of {valid}"
                ) from exc

            conditions = s.get("conditions") or {}
            if not isinstance(conditions, dict):
                raise ValueError(
                    f"{path}: sample {i} has 'conditions' of type "
                    f"{type(conditions).__name__}, expected an object mapping axis "
                    "names to values. A non-dict here fails much later, inside the "
                    "eval report, with an error that doesn't name this file."
                )

            samples.append(
                Sample(
                    path=s["path"],
                    attack_type=attack_type,
                    subject=s["subject"],
                    session=s.get("session", "default"),
                    instrument=s.get("instrument"),
                    conditions={str(k): str(v) for k, v in conditions.items()},
                )
            )
        return cls(samples=samples, root=data.get("root") or "")

    def image_root(self, fallback: Path | str) -> Path:
        """Where this manifest's relative paths actually resolve from.

        The manifest records the tree its images live in, which is not always the
        directory the manifest itself sits in. `tyf capture` writes both to the same
        place, so the distinction never came up; `tyf import-celeba` writes a
        manifest into your data dir while the 625K images stay in the CelebA
        download, and copying them would be absurd.

        Consumers used to assume `--data` was the image root, which meant an
        imported manifest failed on the first image read with a path that didn't
        exist — the manifest was right there and being ignored. Falls back to the
        given directory when the recorded root is absent or stale, so a dataset
        that was moved wholesale still works.
        """
        if self.root:
            root = Path(self.root)
            if root.exists():
                return root
        return Path(fallback)

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
    # Fractions are validated because the failure is silent, not loud: passing 20
    # meaning "20%" clamps to one val group and one test group and still returns a
    # plausible-looking split, so the reported metrics would come from a test set
    # of one subject without anything saying so.
    for label, frac in (("val_fraction", val_fraction), ("test_fraction", test_fraction)):
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"{label} must be in [0, 1), got {frac}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must leave room for training, got "
            f"{val_fraction} + {test_fraction} = {val_fraction + test_fraction}"
        )

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
    # Guarantee at least one group left for training. Falling back to one group
    # each is a real change to what was asked for — with 10 subjects and
    # val_fraction=0.9 it turns 9 val groups into 1 — so it is announced rather
    # than applied silently. A test set that quietly shrank to a single subject
    # still produces confident-looking metrics.
    if n_test + n_val >= n:
        print(
            f"warning: val_fraction={val_fraction} + test_fraction={test_fraction} "
            f"would leave no training {by}s ({n_val} val + {n_test} test of {n} "
            f"{by}s). Using 1 val and 1 test {by} instead; metrics from a "
            f"single-{by} split are indicative at best."
        )
        n_test, n_val = 1, 1

    test_names = set(names[:n_test])
    val_names = set(names[n_test : n_test + n_val])

    train, val, test = [], [], []
    for name in names:
        target = test if name in test_names else val if name in val_names else train
        target.extend(groups[name])

    return train, val, test


def split_fingerprint(
    train: list[Sample], val: list[Sample], test: list[Sample], by: str = "subject"
) -> dict:
    """Record exactly which groups landed in which split.

    Written next to a checkpoint at train time so `tyf eval` can prove it is
    scoring the same held-out data, rather than assuming it. See
    `verify_split_matches` for why assuming is not good enough.
    """
    key = (lambda s: s.subject) if by == "subject" else (lambda s: f"{s.subject}/{s.session}")
    return {
        "by": by,
        "train": sorted({key(s) for s in train}),
        "val": sorted({key(s) for s in val}),
        "test": sorted({key(s) for s in test}),
    }


def verify_split_matches(fingerprint: dict, test: list[Sample], by: str = "subject") -> None:
    """Check a freshly reconstructed test split against the one used at train time.

    This closes a leak that no other check catches. `tyf eval` does not read a
    stored split — it re-derives one from the manifest using `--seed` and
    `--split-by`. Those default to 0 and "subject" regardless of what training
    actually used, so evaluating a model trained with `--seed 7` produces a
    *different* partition, and subjects the model trained on land in the
    "held-out" test set. Every reported number then flatters the model, and
    nothing about the output looks wrong: the split is still internally disjoint,
    so `check_split_integrity` passes and the report reads normally.

    Comparing against the recorded groups is what makes "held-out" verifiable
    rather than a claim about how the command was invoked.
    """
    key = (lambda s: s.subject) if by == "subject" else (lambda s: f"{s.subject}/{s.session}")
    got = sorted({key(s) for s in test})
    expected = sorted(fingerprint.get("test", []))
    if not expected:
        return

    if fingerprint.get("by") != by:
        raise ValueError(
            f"this model was trained with --split-by {fingerprint.get('by')!r} but you "
            f"passed {by!r}. The reconstructed split differs from the trained one."
        )
    if got != expected:
        leaked = sorted(set(got) & set(fingerprint.get("train", [])))
        detail = (
            f" {len(leaked)} of them were TRAINED on: {leaked[:5]}"
            if leaked
            else " no trained groups leaked in, but the partition still differs"
        )
        raise ValueError(
            f"the reconstructed test split does not match the one used for training.\n"
            f"  trained on test {by}s: {expected}\n"
            f"  reconstructed:        {got}\n"
            f"{detail}.\n"
            "Pass the --seed and --split-by that training used; the values are in "
            "train_summary.json under 'split'."
        )


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

    The shape is checked because the wrong one doesn't raise on its own: a 4-channel
    BGRA frame (what some capture backends and PNG reads produce) passes straight
    through the reverse-and-transpose and yields a (4, H, W) tensor, which then
    fails deep inside the model with a shape error that names neither this function
    nor the image that caused it.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) BGR uint8 image, got shape {img.shape}. A 4-channel "
            "(BGRA) or single-channel image must be converted before this point."
        )
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
