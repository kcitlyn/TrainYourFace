"""Convert CelebA-Spoof into a TrainYourFace manifest.

CelebA-Spoof (Zhang et al., ECCV 2020) is 625,537 images from 10,177 subjects, and
unlike CASIA-FASD / Replay-Attack / OULU-NPU / SiW it downloads directly with no
institutional license application. That makes it the only realistic route to a
benchmark number for this project without a signed agreement.

It is also the reason `tyf eval --cross` exists. Training here and testing on your
own webcam captures measures the thing that actually matters — whether the model
survives a camera and a room it has never seen — instead of the thing every
published PAD model looks good at.

LICENSE
-------
Non-commercial research and educational use only, per the dataset's own terms.
Nothing from it is redistributed by this project: the converter reads a copy you
downloaded yourself and writes only a manifest of relative paths. Cite the paper
if you publish numbers from it.

THE ANNOTATION FORMAT
---------------------
`metas/intra_test/{train,test}_label.json` maps a relative image path to a
44-element list:

    [0:40]  the 40 CelebA face attributes, encoded +1/-1 (see FAIRNESS_ATTRIBUTES)
    [40]    spoof type    0=Live 1=Photo 2=Poster 3=A4 4=FaceMask
                          5=UpperBodyMask 6=RegionMask 7=PC 8=Pad 9=Phone 10=3DMask
    [41]    illumination  0=Live 1=Normal 2=Strong 3=Back 4=Dark
    [42]    environment   0=Live 1=Indoor 2=Outdoor
    [43]    live/spoof    0=live 1=spoof

Code 0 in indices 40-42 is reserved for live images, so those attributes only
carry information on spoof samples. That asymmetry is why `conditions` is left
empty for bona-fide samples rather than recorded as the string "live" — writing
"live" into an illumination field would create a condition bucket that is really
just the class label, and stratifying by it would report the trivial result that
live images are classified as live.

SUBJECTS COME FROM THE PATH
---------------------------
Paths look like `Data/train/1234/live/000001.png`, where `1234` is the CelebA
identity. That component is the subject ID, and extracting it is load-bearing
rather than cosmetic: it is what makes `subject_disjoint_split` able to guarantee
no identity spans train and test. If the path layout ever changes and this parser
silently falls back to one subject per file, every split becomes a random
per-image split and the leakage trap in `dataset.py` reopens. So a failure to
find an identity component is an error, not a default.
"""

from __future__ import annotations

import json
from pathlib import Path

from trainyourface.core.contracts import AttackType
from trainyourface.liveness.dataset import DatasetManifest, Sample

# Spoof-type code -> our AttackType vocabulary.
#
# The mapping is lossy on purpose, because our enum follows the ISO/IEC 30107-3
# categories and CelebA-Spoof's 11 classes are finer-grained. Where it collapses
# categories it collapses them by PRESENTATION MEDIUM, which is what a PAD model
# actually sees:
#   Photo/Poster/A4     -> PRINT   (all ink on paper; the cue is halftone + flatness)
#   PC/Pad/Phone        -> REPLAY  (all emissive screens; the cue is moiré + backlight)
#   FaceMask/UpperBody  -> CUTOUT  (paper masks with regions removed)
#   3DMask              -> MASK_3D
#
# The original fine-grained code is preserved in `conditions["spoof_type"]`, so
# collapsing here loses nothing — per-medium APCER is still reportable, and the
# finer breakdown is available for anyone who wants it.
SPOOF_TYPE: dict[int, AttackType] = {
    0: AttackType.BONA_FIDE,
    1: AttackType.PRINT,
    2: AttackType.PRINT,
    3: AttackType.PRINT,
    4: AttackType.CUTOUT,
    5: AttackType.CUTOUT,
    6: AttackType.CUTOUT,
    7: AttackType.REPLAY,
    8: AttackType.REPLAY,
    9: AttackType.REPLAY,
    10: AttackType.MASK_3D,
}

SPOOF_TYPE_NAME: dict[int, str] = {
    0: "live",
    1: "photo",
    2: "poster",
    3: "a4",
    4: "face_mask",
    5: "upper_body_mask",
    6: "region_mask",
    7: "pc",
    8: "pad",
    9: "phone",
    10: "mask_3d",
}

ILLUMINATION: dict[int, str] = {1: "normal", 2: "strong", 3: "back", 4: "dark"}
ENVIRONMENT: dict[int, str] = {1: "indoor", 2: "outdoor"}

# The 40 CelebA face attributes, in the canonical order of the dataset's own
# list_attr_celeba.txt. Indices 0-39 of every label vector.
#
# Unlike the three spoof attributes above, these are present on LIVE images, which
# is what makes a fairness audit possible at all: the harm a PAD model does to
# real people is measured by BPCER (genuine users wrongly rejected), and BPCER
# needs labels on bona-fide samples.
CELEBA_ATTRIBUTES: tuple[str, ...] = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)

# The subset worth auditing for disparate impact, and the reasoning for the cut is
# the point rather than an implementation detail.
#
# INCLUDED because a PAD model treating these groups differently is a real harm:
#   Male, Young, Pale_Skin  -- demographic proxies (see the caveat below)
#   Eyeglasses, Wearing_Hat -- accessories a user cannot reasonably be asked to
#                              remove at every unlock, and which occlude the face
#   Heavy_Makeup, Wearing_Lipstick -- correlate strongly with gender presentation
#                              and alter skin texture, which is the PAD cue itself
#   Bald, Gray_Hair, No_Beard -- age and grooming proxies
#
# EXCLUDED deliberately:
#   Attractive -- a subjective crowd annotation. Reporting a rate "by
#                 attractiveness" would treat it as a real category, which is a
#                 claim this project has no business making.
#   Blurry     -- an image-quality attribute, not a property of the person. It
#                 belongs in a robustness analysis, not a fairness one.
#
# THE CAVEAT THAT MATTERS: `Pale_Skin` is a binary crowd-sourced annotation, NOT a
# validated skin-tone measurement. It is not Fitzpatrick, not the Monk scale, and
# it collapses a continuum into one bit decided by an annotator. A disparity found
# along it is a signal worth investigating, not a measured skin-tone bias, and
# reporting it as the latter would be exactly the kind of overclaim the rest of
# this project refuses. The honest version of this audit needs Monk-scale labels
# that CelebA-Spoof does not have.
FAIRNESS_ATTRIBUTES: tuple[str, ...] = (
    "Male",
    "Young",
    "Pale_Skin",
    "Eyeglasses",
    "Wearing_Hat",
    "Heavy_Makeup",
    "Wearing_Lipstick",
    "Bald",
    "Gray_Hair",
    "No_Beard",
)

# The label vector must be at least this long to index [43]. Checked rather than
# assumed: a truncated or differently-versioned annotation file would otherwise
# raise IndexError deep in the loop, naming a list index instead of a file.
LABEL_LEN = 44


def _subject_from_path(rel_path: str) -> str:
    """Pull the CelebA identity out of `Data/train/1234/live/000001.png`.

    Returns the first path component that is all digits. The identity directory is
    the only numeric component in the documented layout — `Data`, `train`, `live`,
    and `spoof` are all words, and the filename has an extension.
    """
    for part in Path(rel_path).parts[:-1]:
        if part.isdigit():
            return part
    raise ValueError(
        f"no numeric identity directory in {rel_path!r}. Expected a CelebA-Spoof "
        "layout like 'Data/train/1234/live/000001.png'; without the identity the "
        "split cannot be subject-disjoint, so this is refused rather than guessed."
    )


def parse_label(vec: list, attributes: bool = False) -> tuple[AttackType, dict[str, str]]:
    """Turn one 44-element annotation vector into (attack_type, conditions).

    Raises ValueError on a vector this code doesn't understand, rather than
    defaulting. A mis-parsed label silently mislabels training data, and a PAD
    model trained on flipped labels still trains — it just reports nonsense.

    Args:
        attributes: also record the CelebA face attributes in FAIRNESS_ATTRIBUTES,
            prefixed `attr:`. Prefixed rather than merged flat because these mean
            something different from capture conditions — they describe the PERSON,
            are present on live images, and are therefore the axis a BPCER
            disparity is measured along. Merging them would put "was the subject
            wearing glasses" in the same namespace as "was the screen backlit".
    """
    if len(vec) < LABEL_LEN:
        raise ValueError(
            f"expected a {LABEL_LEN}-element label vector, got {len(vec)}. This "
            "converter targets the CelebA-Spoof intra_test annotation format."
        )

    spoof_code = int(vec[40])
    is_spoof = bool(int(vec[43]))

    if spoof_code not in SPOOF_TYPE:
        raise ValueError(f"unknown spoof-type code {spoof_code}; expected 0-10")

    attack = SPOOF_TYPE[spoof_code]

    # Cross-check the two label fields against each other. They encode the same
    # fact at different granularities, so a disagreement means the file is not the
    # format this parser expects — and trusting either one alone would mislabel
    # data silently. Cheap check, catches a whole class of format drift.
    if is_spoof != attack.is_attack:
        raise ValueError(
            f"label vector disagrees with itself: live/spoof flag says "
            f"{'spoof' if is_spoof else 'live'} but spoof type is "
            f"{SPOOF_TYPE_NAME[spoof_code]!r}. Refusing to guess which is right."
        )

    conditions: dict[str, str] = {}
    if is_spoof:
        # Only meaningful for spoofs — see the module docstring on why live
        # samples get an empty dict instead of "live".
        conditions["spoof_type"] = SPOOF_TYPE_NAME[spoof_code]
        if (illum := ILLUMINATION.get(int(vec[41]))) is not None:
            conditions["illumination"] = illum
        if (env := ENVIRONMENT.get(int(vec[42]))) is not None:
            conditions["environment"] = env

    if attributes:
        # CelebA encodes these as +1 / -1, not 1 / 0. Testing `> 0` rather than
        # truthiness matters: -1 is truthy in Python, so a naive bool() would
        # label every attribute present on every face and produce a fairness
        # report with one bucket per axis and no disparity anywhere — a clean
        # bill of health manufactured by a sign error.
        for name in FAIRNESS_ATTRIBUTES:
            raw = vec[CELEBA_ATTRIBUTES.index(name)]
            conditions[f"attr:{name}"] = "yes" if int(raw) > 0 else "no"

    return attack, conditions


def convert(
    root: Path | str,
    label_file: Path | str | None = None,
    limit: int | None = None,
    session: str = "celeba",
    attributes: bool = False,
    log=print,
) -> DatasetManifest:
    """Build a manifest from a downloaded CelebA-Spoof tree.

    Args:
        root: the directory containing `Data/` and `metas/`.
        label_file: annotation JSON. Defaults to metas/intra_test/train_label.json.
        limit: keep only this many images. 625K images is more than a laptop wants
            to train on, and sampling is how you get a run that finishes — but it
            samples by SUBJECT, not by image, so a limited manifest is still
            subject-disjoint-splittable and still has whole identities in it.
        session: recorded on every sample, so a later cross-dataset eval can tell
            CelebA samples from your own captures by session name.
        attributes: record the CelebA face attributes for a fairness audit. Off by
            default because it adds ~10 keys to every sample, which is dead weight
            in a manifest nobody will audit.

    Skipped images are counted and reported rather than silently dropped: a
    converter that quietly ignores half the dataset produces a manifest whose size
    is a mystery.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no such directory: {root}")

    if label_file is None:
        label_file = root / "metas" / "intra_test" / "train_label.json"
    label_file = Path(label_file)
    if not label_file.exists():
        raise FileNotFoundError(
            f"no annotation file at {label_file}. Expected the CelebA-Spoof "
            "'metas/intra_test/train_label.json' that ships with the download; "
            "pass --label-file if yours is elsewhere."
        )

    labels = json.loads(label_file.read_text())
    if not isinstance(labels, dict) or not labels:
        raise ValueError(
            f"{label_file} is not a non-empty JSON object mapping image paths to label vectors."
        )
    log(f"read {len(labels)} annotations from {label_file.name}")

    # Group by subject BEFORE limiting, so `limit` can cut whole identities.
    by_subject: dict[str, list[tuple[str, list]]] = {}
    bad_path = bad_label = missing = 0
    for rel, vec in labels.items():
        try:
            subject = _subject_from_path(rel)
        except ValueError:
            bad_path += 1
            continue
        by_subject.setdefault(subject, []).append((rel, vec))

    if not by_subject:
        raise ValueError(
            f"no usable identity directories found in {label_file}. All "
            f"{len(labels)} paths failed to parse, so the layout is not the one "
            "this converter expects."
        )

    samples: list[Sample] = []
    kept_subjects = 0
    for subject in sorted(by_subject, key=int):
        if limit is not None and len(samples) >= limit:
            break
        subject_samples: list[Sample] = []
        for rel, vec in by_subject[subject]:
            if not (root / rel).exists():
                missing += 1
                continue
            try:
                attack, conditions = parse_label(vec, attributes=attributes)
            except ValueError:
                bad_label += 1
                continue
            subject_samples.append(
                Sample(
                    path=rel,
                    attack_type=attack,
                    subject=subject,
                    session=session,
                    instrument=conditions.get("spoof_type"),
                    conditions=conditions,
                )
            )
        if subject_samples:
            samples.extend(subject_samples)
            kept_subjects += 1

    for count, what in (
        (bad_path, "paths with no identity directory"),
        (bad_label, "unparseable label vectors"),
        (missing, "annotated images not present on disk"),
    ):
        if count:
            log(f"  skipped {count} {what}")

    if not samples:
        raise ValueError(
            "converted 0 samples. The annotation file parsed but none of the "
            f"images it names exist under {root} — check that --root points at "
            "the directory CONTAINING Data/, not at Data/ itself."
        )

    manifest = DatasetManifest(samples=samples, root=str(root))
    log(f"built {len(samples)} samples across {kept_subjects} subjects")
    return manifest
