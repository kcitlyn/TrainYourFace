"""Presentation Attack Detection metrics, per ISO/IEC 30107-3.

Why this module exists and why it is the most important file in the repo:

Reporting "97% accuracy" on a spoof-detection dataset is close to meaningless.
PAD test sets are usually attack-heavy — a 1:3 bona-fide:attack ratio is normal —
so a model that flags *everything* as an attack scores 75% and is useless. Worse,
accuracy collapses two errors with completely different costs into one number:
letting an attacker in, versus locking out the real user.

The standard vocabulary instead:

  APCER (Attack Presentation Classification Error Rate)
      fraction of attacks that were wrongly accepted as real.
      This is the security failure. Reported PER ATTACK TYPE, and the headline
      figure is the WORST category, not the mean — see `worst_apcer`.

  BPCER (Bona-fide Presentation Classification Error Rate)
      fraction of real users wrongly rejected as attacks.
      This is the usability failure.

  ACER (Average Classification Error Rate)
      (APCER + BPCER) / 2. Convenient single number, but it hides the tradeoff,
      so we always report the components next to it.

Because APCER and BPCER trade off against each other through the decision
threshold, a single operating point says little. So we also compute:

  EER   — the threshold where the two rates are equal; a threshold-free summary.
  BPCER@APCER=x — the usability cost of pinning security at x. This is how the
      biometrics literature actually compares systems, and it is the number an
      interviewer will ask for.

Everything here is pure NumPy over (score, label) arrays: no torch, no model, no
I/O. That keeps it unit-testable against hand-computed fixtures, which matters
because a metrics bug produces confidently wrong claims rather than a crash.

Convention throughout: `scores` are SPOOF probabilities in [0, 1] — higher means
more attack-like — and a sample is classified as an attack when
`score >= threshold`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from trainyourface.core.contracts import AttackType


class OperatingPoint(BaseModel):
    """PAD error rates at one specific decision threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    # Per-attack-type APCER. Keyed by AttackType value so a JSON dump of this
    # model is directly readable in a report.
    apcer_by_type: dict[str, float] = Field(default_factory=dict)
    bpcer: float
    n_bona_fide: int
    n_attack: int

    @property
    def worst_apcer(self) -> float:
        """The highest APCER across attack types — the honest security number.

        An attacker picks their attack; they are not assigned a random one. So the
        system's real security level is set by its weakest category, which makes
        the max the correct aggregate and the mean actively misleading.
        """
        if not self.apcer_by_type:
            return 0.0
        return max(self.apcer_by_type.values())

    @property
    def mean_apcer(self) -> float:
        """Unweighted mean APCER across types. Reported for comparability only.

        Some papers report this. We surface it so numbers can be compared against
        published tables, but `worst_apcer` is what this project headlines.
        """
        if not self.apcer_by_type:
            return 0.0
        return float(np.mean(list(self.apcer_by_type.values())))

    @property
    def acer(self) -> float:
        """ACER using the worst-case APCER."""
        return (self.worst_apcer + self.bpcer) / 2.0


class PADReport(BaseModel):
    """Full evaluation result: threshold-free summaries plus chosen operating points."""

    model_config = ConfigDict(frozen=True)

    eer: float = Field(description="Equal Error Rate — threshold-independent summary.")
    eer_threshold: float
    auc: float = Field(description="Area under the ROC curve.")
    # BPCER at fixed APCER targets — the standard comparison table.
    bpcer_at_apcer: dict[str, float] = Field(default_factory=dict)
    operating_points: list[OperatingPoint] = Field(default_factory=list)
    n_bona_fide: int = 0
    n_attack: int = 0
    attack_types_present: list[str] = Field(default_factory=list)


def _validate(scores: np.ndarray, is_attack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    is_attack = np.asarray(is_attack, dtype=bool).ravel()
    if scores.shape != is_attack.shape:
        raise ValueError(f"scores {scores.shape} and labels {is_attack.shape} must align")
    if scores.size == 0:
        raise ValueError("cannot compute PAD metrics on an empty score array")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or inf")
    return scores, is_attack


def apcer_bpcer(
    scores: Sequence[float] | np.ndarray,
    attack_types: Sequence[AttackType] | Sequence[str],
    threshold: float,
) -> OperatingPoint:
    """Compute per-type APCER and BPCER at one threshold.

    Args:
        scores: spoof probability per sample, higher = more attack-like.
        attack_types: per-sample category; BONA_FIDE marks a genuine sample.
        threshold: a sample is called an attack when score >= threshold.

    A note on the degenerate case: if a split contains no samples of some attack
    type, that type is simply absent from `apcer_by_type` rather than recorded as
    0.0. Recording 0.0 would claim perfect performance against an attack we never
    tested, which is exactly the kind of flattering-but-false number this module
    exists to prevent.
    """
    types = [AttackType(t) if not isinstance(t, AttackType) else t for t in attack_types]
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    if scores_arr.size != len(types):
        raise ValueError(f"got {scores_arr.size} scores for {len(types)} labels")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores contain NaN or inf")

    types_arr = np.array([t.value for t in types])
    bona_mask = types_arr == AttackType.BONA_FIDE.value

    # BPCER: genuine samples wrongly flagged as attacks.
    n_bona = int(bona_mask.sum())
    if n_bona:
        bpcer = float((scores_arr[bona_mask] >= threshold).mean())
    else:
        bpcer = 0.0

    # APCER per type: attacks wrongly accepted as genuine.
    apcer_by_type: dict[str, float] = {}
    for t in AttackType:
        if t is AttackType.BONA_FIDE:
            continue
        mask = types_arr == t.value
        if not mask.any():
            continue  # untested category — omit rather than claim 0.0
        apcer_by_type[t.value] = float((scores_arr[mask] < threshold).mean())

    return OperatingPoint(
        threshold=float(threshold),
        apcer_by_type=apcer_by_type,
        bpcer=bpcer,
        n_bona_fide=n_bona,
        n_attack=int((~bona_mask).sum()),
    )


def roc_curve(
    scores: Sequence[float] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC over all achievable thresholds, computed without sklearn.

    Returns (thresholds, apcer, bpcer) — apcer descending as threshold rises.
    Implemented directly so the core install needs no sklearn and so the math is
    inspectable rather than delegated.
    """
    scores_arr, attack_mask = _validate(np.asarray(scores), np.asarray(is_attack))

    # Candidate thresholds: every distinct score, plus sentinels just outside the
    # range so the curve reaches both (0,1) and (1,0) corners.
    uniq = np.unique(scores_arr)
    thresholds = np.concatenate([[-np.inf], uniq, [np.inf]])

    attack_scores = scores_arr[attack_mask]
    bona_scores = scores_arr[~attack_mask]

    # Vectorized over thresholds: (n_thresh, 1) vs (1, n_samples) broadcast.
    if attack_scores.size:
        apcer = (attack_scores[None, :] < thresholds[:, None]).mean(axis=1)
    else:
        apcer = np.zeros_like(thresholds)
    if bona_scores.size:
        bpcer = (bona_scores[None, :] >= thresholds[:, None]).mean(axis=1)
    else:
        bpcer = np.zeros_like(thresholds)

    return thresholds, apcer, bpcer


def equal_error_rate(
    scores: Sequence[float] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
) -> tuple[float, float]:
    """Equal Error Rate and the threshold achieving it.

    EER is where APCER == BPCER. With finite samples the two curves usually cross
    between adjacent thresholds rather than meeting exactly, so we take the
    threshold minimizing |APCER - BPCER| and average the two rates there — the
    standard discrete approximation.
    """
    thresholds, apcer, bpcer = roc_curve(scores, is_attack)
    idx = int(np.argmin(np.abs(apcer - bpcer)))
    eer = float((apcer[idx] + bpcer[idx]) / 2.0)
    thr = float(thresholds[idx])
    # The sentinel thresholds are ±inf; report a usable finite threshold instead.
    if not np.isfinite(thr):
        finite = np.asarray(scores, dtype=np.float64)
        thr = float(finite.min() if thr == -np.inf else finite.max())
    return eer, thr


def auc_score(
    scores: Sequence[float] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
) -> float:
    """ROC AUC via the Mann-Whitney U statistic, with correct tie handling.

    Using the rank-based identity rather than trapezoidal integration because it
    is exact and handles tied scores by averaging ranks — quantized INT8 models
    produce many ties, and naive integration mishandles them.
    """
    scores_arr, attack_mask = _validate(np.asarray(scores), np.asarray(is_attack))
    n_attack = int(attack_mask.sum())
    n_bona = int((~attack_mask).sum())
    if n_attack == 0 or n_bona == 0:
        raise ValueError("AUC needs at least one bona-fide and one attack sample")

    # Average ranks over ties.
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    ranks = np.empty(len(scores_arr), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    rank_sum_attack = ranks[attack_mask].sum()
    u = rank_sum_attack - n_attack * (n_attack + 1) / 2.0
    return float(u / (n_attack * n_bona))


def bpcer_at_apcer(
    scores: Sequence[float] | np.ndarray,
    is_attack: Sequence[bool] | np.ndarray,
    target_apcer: float,
) -> tuple[float, float]:
    """BPCER when the threshold is set so APCER <= target. Returns (bpcer, threshold).

    "How many real users do we inconvenience to hold attack acceptance at 1%?" —
    the question that actually decides whether a PAD model is deployable.
    """
    if not 0.0 <= target_apcer <= 1.0:
        raise ValueError(f"target_apcer must be in [0, 1], got {target_apcer}")
    thresholds, apcer, bpcer = roc_curve(scores, is_attack)

    feasible = np.flatnonzero(apcer <= target_apcer)
    if feasible.size == 0:
        # Even the strictest threshold can't reach the target; report the best
        # achievable rather than silently returning a number that implies success.
        return 1.0, float(thresholds[int(np.argmin(apcer))])

    # Among thresholds meeting the security target, choose the kindest to users.
    best = feasible[int(np.argmin(bpcer[feasible]))]
    thr = float(thresholds[best])
    if not np.isfinite(thr):
        arr = np.asarray(scores, dtype=np.float64)
        thr = float(arr.min() if thr == -np.inf else arr.max())
    return float(bpcer[best]), thr


def evaluate(
    scores: Sequence[float] | np.ndarray,
    attack_types: Sequence[AttackType] | Sequence[str],
    thresholds: Sequence[float] | None = None,
    apcer_targets: Sequence[float] = (0.01, 0.05, 0.10),
) -> PADReport:
    """Produce the full PAD report. This is what `tyf eval` prints and saves."""
    types = [AttackType(t) if not isinstance(t, AttackType) else t for t in attack_types]
    is_attack = np.array([t.is_attack for t in types], dtype=bool)
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()

    if is_attack.all() or (~is_attack).all():
        raise ValueError(
            "evaluation split must contain both bona-fide and attack samples; "
            "PAD metrics are undefined otherwise"
        )

    eer, eer_thr = equal_error_rate(scores_arr, is_attack)
    auc = auc_score(scores_arr, is_attack)

    targets: dict[str, float] = {}
    for t in apcer_targets:
        b, _ = bpcer_at_apcer(scores_arr, is_attack, t)
        targets[f"apcer<={t:.2f}"] = b

    # Default operating points: the EER threshold plus a neutral 0.5, so the
    # report always shows both the tuned and the naive choice.
    if thresholds is None:
        thresholds = sorted({round(eer_thr, 6), 0.5})

    points = [apcer_bpcer(scores_arr, types, thr) for thr in thresholds]

    present = sorted({t.value for t in types if t.is_attack})
    return PADReport(
        eer=eer,
        eer_threshold=eer_thr,
        auc=auc,
        bpcer_at_apcer=targets,
        operating_points=points,
        n_bona_fide=int((~is_attack).sum()),
        n_attack=int(is_attack.sum()),
        attack_types_present=present,
    )
