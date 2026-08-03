"""Tests for PAD metrics.

These are checked against values computed BY HAND, not against another library's
output. That's deliberate: a metrics bug here wouldn't crash, it would produce a
confidently wrong claim in the README. Hand-computed fixtures are the only way to
catch a plausible-looking-but-wrong implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.contracts import AttackType
from trainyourface.eval.metrics import (
    apcer_bpcer,
    auc_score,
    bpcer_at_apcer,
    equal_error_rate,
    evaluate,
    roc_curve,
)

BF = AttackType.BONA_FIDE
PR = AttackType.PRINT
RP = AttackType.REPLAY


class TestApcerBpcer:
    def test_perfect_separation(self):
        """Bona-fide all score 0.0, attacks all score 1.0 -> zero error at 0.5."""
        scores = [0.0, 0.0, 1.0, 1.0]
        types = [BF, BF, PR, PR]
        op = apcer_bpcer(scores, types, threshold=0.5)
        assert op.bpcer == 0.0
        assert op.apcer_by_type == {"print": 0.0}
        assert op.acer == 0.0

    def test_hand_computed_mixed(self):
        """Worked by hand.

        threshold = 0.5, attack iff score >= 0.5

        bona-fide scores: 0.1, 0.2, 0.9, 0.4  -> 0.9 is >= 0.5, wrongly rejected
            BPCER = 1/4 = 0.25
        print scores: 0.8, 0.3   -> 0.3 is < 0.5, wrongly accepted
            APCER[print] = 1/2 = 0.50
        replay scores: 0.7, 0.6, 0.2, 0.1 -> 0.2 and 0.1 wrongly accepted
            APCER[replay] = 2/4 = 0.50
        """
        scores = [0.1, 0.2, 0.9, 0.4, 0.8, 0.3, 0.7, 0.6, 0.2, 0.1]
        types = [BF, BF, BF, BF, PR, PR, RP, RP, RP, RP]
        op = apcer_bpcer(scores, types, threshold=0.5)

        assert op.bpcer == pytest.approx(0.25)
        assert op.apcer_by_type["print"] == pytest.approx(0.50)
        assert op.apcer_by_type["replay"] == pytest.approx(0.50)
        assert op.n_bona_fide == 4
        assert op.n_attack == 6
        # worst == mean == 0.5 here, so ACER = (0.5 + 0.25)/2
        assert op.acer == pytest.approx(0.375)

    def test_worst_apcer_not_mean(self):
        """The headline number must be the WORST attack type, not the average.

        This is the core anti-flattery guarantee: strong on print, useless on
        replay must NOT average out to "decent".
        """
        # print: both caught (score 0.9 >= 0.5) -> APCER 0.0
        # replay: both missed (score 0.1 < 0.5) -> APCER 1.0
        scores = [0.1, 0.9, 0.9, 0.1, 0.1]
        types = [BF, PR, PR, RP, RP]
        op = apcer_bpcer(scores, types, threshold=0.5)

        assert op.apcer_by_type["print"] == pytest.approx(0.0)
        assert op.apcer_by_type["replay"] == pytest.approx(1.0)
        assert op.worst_apcer == pytest.approx(1.0)
        assert op.mean_apcer == pytest.approx(0.5)
        # ACER must be driven by the failure, not diluted by the success.
        assert op.acer == pytest.approx(0.5)

    def test_untested_attack_type_is_omitted_not_zero(self):
        """An attack type with no samples must be ABSENT, never recorded as 0.0.

        Recording 0.0 would claim perfect defense against something never tested.
        """
        scores = [0.1, 0.9]
        types = [BF, PR]
        op = apcer_bpcer(scores, types, threshold=0.5)
        assert set(op.apcer_by_type) == {"print"}
        assert "replay" not in op.apcer_by_type
        assert "mask_3d" not in op.apcer_by_type

    def test_threshold_boundary_is_inclusive(self):
        """score == threshold counts as an attack (>= convention)."""
        # A bona-fide sample exactly at threshold is rejected -> BPCER 1.0
        op = apcer_bpcer([0.5], [BF], threshold=0.5)
        assert op.bpcer == pytest.approx(1.0)
        # An attack exactly at threshold is caught -> APCER 0.0
        op = apcer_bpcer([0.5], [PR], threshold=0.5)
        assert op.apcer_by_type["print"] == pytest.approx(0.0)

    def test_accepts_raw_strings(self):
        op = apcer_bpcer([0.1, 0.9], ["bona_fide", "print"], threshold=0.5)
        assert op.apcer_by_type["print"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="scores for"):
            apcer_bpcer([0.1, 0.2], [BF], threshold=0.5)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            apcer_bpcer([0.1, float("nan")], [BF, PR], threshold=0.5)


class TestRocAndAuc:
    def test_auc_perfect(self):
        scores = [0.0, 0.1, 0.9, 1.0]
        is_attack = [False, False, True, True]
        assert auc_score(scores, is_attack) == pytest.approx(1.0)

    def test_auc_inverted(self):
        """Perfectly backwards model -> AUC 0.0."""
        scores = [1.0, 0.9, 0.1, 0.0]
        is_attack = [False, False, True, True]
        assert auc_score(scores, is_attack) == pytest.approx(0.0)

    def test_auc_all_ties_is_half(self):
        """Every score identical -> no discrimination -> AUC 0.5.

        Tie handling matters: quantized INT8 models produce many exact ties, and
        an implementation that mishandles them reports inflated AUC.
        """
        scores = [0.5, 0.5, 0.5, 0.5]
        is_attack = [False, False, True, True]
        assert auc_score(scores, is_attack) == pytest.approx(0.5)

    def test_auc_hand_computed_with_ties(self):
        """Hand-computed via Mann-Whitney U with averaged ranks.

        scores:    bona = [0.2, 0.5], attack = [0.5, 0.8]
        sorted:    0.2(b), 0.5(b), 0.5(a), 0.8(a)
        ranks:     0.2 -> 1
                   0.5, 0.5 tie at positions 2,3 -> both get 2.5
                   0.8 -> 4
        attack rank sum = 2.5 + 4 = 6.5
        U = 6.5 - n_a(n_a+1)/2 = 6.5 - 3 = 3.5
        AUC = U / (n_a * n_b) = 3.5 / 4 = 0.875
        """
        scores = [0.2, 0.5, 0.5, 0.8]
        is_attack = [False, False, True, True]
        assert auc_score(scores, is_attack) == pytest.approx(0.875)

    def test_auc_single_class_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            auc_score([0.1, 0.2], [True, True])

    def test_roc_spans_both_corners(self):
        """Sentinel thresholds must drive the curve to both extremes."""
        scores = [0.2, 0.4, 0.6, 0.8]
        is_attack = [False, False, True, True]
        thr, apcer, bpcer = roc_curve(scores, is_attack)
        # At threshold -inf everything is called an attack: APCER 0, BPCER 1.
        assert apcer[0] == pytest.approx(0.0)
        assert bpcer[0] == pytest.approx(1.0)
        # At threshold +inf nothing is: APCER 1, BPCER 0.
        assert apcer[-1] == pytest.approx(1.0)
        assert bpcer[-1] == pytest.approx(0.0)

    def test_roc_monotonic(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        is_attack = rng.random(200) > 0.5
        thr, apcer, bpcer = roc_curve(scores, is_attack)
        # As the threshold rises, more attacks slip through and fewer genuine
        # users are rejected — both curves must be monotonic.
        assert np.all(np.diff(apcer) >= -1e-12)
        assert np.all(np.diff(bpcer) <= 1e-12)


class TestEER:
    def test_eer_perfect_is_zero(self):
        eer, thr = equal_error_rate([0.0, 0.1, 0.9, 1.0], [False, False, True, True])
        assert eer == pytest.approx(0.0)
        assert np.isfinite(thr)

    def test_eer_random_is_near_half(self):
        """A coin-flip model should land near 0.5 EER."""
        rng = np.random.default_rng(42)
        n = 4000
        scores = rng.random(n)
        is_attack = rng.random(n) > 0.5
        eer, _ = equal_error_rate(scores, is_attack)
        assert 0.42 < eer < 0.58

    def test_eer_threshold_always_finite(self):
        """The returned threshold must be usable, never ±inf."""
        eer, thr = equal_error_rate([0.3, 0.3, 0.3, 0.3], [False, False, True, True])
        assert np.isfinite(thr)


class TestBpcerAtApcer:
    def test_perfect_model_costs_nothing(self):
        scores = [0.0, 0.0, 1.0, 1.0]
        is_attack = [False, False, True, True]
        bpcer, thr = bpcer_at_apcer(scores, is_attack, 0.01)
        assert bpcer == pytest.approx(0.0)

    def test_hand_computed_tradeoff(self):
        """bona = [0.1,0.2,0.3,0.4], attack = [0.5,0.6,0.7,0.8].

        Perfectly separable, so pinning APCER at 0 costs no usability.
        """
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        is_attack = [False] * 4 + [True] * 4
        bpcer, thr = bpcer_at_apcer(scores, is_attack, 0.0)
        assert bpcer == pytest.approx(0.0)
        assert 0.4 < thr <= 0.5

    def test_unreachable_target_reports_failure(self):
        """If the target APCER is unreachable, report BPCER 1.0 rather than a lie."""
        # Inverted model: attacks score LOWER than genuine, so low APCER is
        # only achievable by rejecting essentially everyone.
        scores = [0.9, 0.9, 0.1, 0.1]
        is_attack = [False, False, True, True]
        bpcer, _ = bpcer_at_apcer(scores, is_attack, 0.0)
        assert bpcer == pytest.approx(1.0)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bpcer_at_apcer([0.1, 0.9], [False, True], 1.5)


class TestEvaluate:
    def test_full_report(self):
        rng = np.random.default_rng(7)
        # A decent-but-imperfect model: genuine low, attacks high, with overlap.
        bona = rng.normal(0.3, 0.15, 150).clip(0, 1)
        print_a = rng.normal(0.7, 0.15, 80).clip(0, 1)
        replay_a = rng.normal(0.65, 0.2, 70).clip(0, 1)

        scores = np.concatenate([bona, print_a, replay_a])
        types = [BF] * 150 + [PR] * 80 + [RP] * 70

        rep = evaluate(scores, types)

        assert rep.n_bona_fide == 150
        assert rep.n_attack == 150
        assert set(rep.attack_types_present) == {"print", "replay"}
        assert 0.0 < rep.eer < 0.5
        assert 0.5 < rep.auc < 1.0
        assert set(rep.bpcer_at_apcer) == {"apcer<=0.01", "apcer<=0.05", "apcer<=0.10"}
        assert len(rep.operating_points) >= 1
        # Tightening the security target can only cost more usability.
        assert rep.bpcer_at_apcer["apcer<=0.01"] >= rep.bpcer_at_apcer["apcer<=0.10"] - 1e-9

    def test_single_class_split_raises(self):
        """A split with no genuine samples must fail loudly, not report a number."""
        with pytest.raises(ValueError, match="both bona-fide and attack"):
            evaluate([0.1, 0.9], [PR, RP])

    def test_report_is_json_serializable(self):
        """Reports get written to disk and pasted into the README."""
        rep = evaluate([0.1, 0.2, 0.8, 0.9], [BF, BF, PR, RP])
        blob = rep.model_dump_json()
        assert "eer" in blob
        assert "print" in blob
