"""Tests for PAD dataset splitting.

The leakage tests here are the highest-value tests in the repo. A subject leak
doesn't crash anything — it just inflates every reported metric toward 100% and
makes the whole evaluation meaningless. So the guarantee is tested directly rather
than assumed.
"""

from __future__ import annotations

import pytest

from trainyourface.core.contracts import AttackType
from trainyourface.liveness.dataset import (
    DatasetManifest,
    Sample,
    check_split_integrity,
    class_weights,
    subject_disjoint_split,
)

BF = AttackType.BONA_FIDE
PR = AttackType.PRINT
RP = AttackType.REPLAY


def build_samples(n_subjects: int, per_class: int = 5, sessions: int = 1) -> list[Sample]:
    out: list[Sample] = []
    for s in range(n_subjects):
        for sess in range(sessions):
            for atk in (BF, PR, RP):
                for i in range(per_class):
                    out.append(
                        Sample(
                            path=f"s{s}_sess{sess}_{atk.value}_{i}.png",
                            attack_type=atk,
                            subject=f"subj{s}",
                            session=f"sess{sess}",
                        )
                    )
    return out


class TestSubjectDisjointSplit:
    def test_no_subject_appears_in_two_splits(self):
        """The core anti-leakage guarantee."""
        samples = build_samples(10)
        train, val, test = subject_disjoint_split(samples, seed=0)

        s_train = {s.subject for s in train}
        s_val = {s.subject for s in val}
        s_test = {s.subject for s in test}

        assert not (s_train & s_val)
        assert not (s_train & s_test)
        assert not (s_val & s_test)

    def test_every_sample_is_used_exactly_once(self):
        samples = build_samples(10)
        train, val, test = subject_disjoint_split(samples, seed=0)
        assert len(train) + len(val) + len(test) == len(samples)
        paths = [s.path for s in train + val + test]
        assert len(paths) == len(set(paths))

    def test_deterministic_for_a_given_seed(self):
        samples = build_samples(10)
        a = subject_disjoint_split(samples, seed=7)
        b = subject_disjoint_split(samples, seed=7)
        for pa, pb in zip(a, b, strict=True):
            assert [s.path for s in pa] == [s.path for s in pb]

    def test_different_seeds_give_different_splits(self):
        samples = build_samples(12)
        a = subject_disjoint_split(samples, seed=1)
        b = subject_disjoint_split(samples, seed=99)
        assert {s.subject for s in a[2]} != {s.subject for s in b[2]}

    def test_session_level_split_is_stricter(self):
        """Splitting by session must not put the same subject+session in two splits."""
        samples = build_samples(4, sessions=3)
        train, val, test = subject_disjoint_split(samples, seed=0, by="session")

        def keys(part):
            return {f"{s.subject}/{s.session}" for s in part}

        assert not (keys(train) & keys(val))
        assert not (keys(train) & keys(test))
        assert not (keys(val) & keys(test))

    def test_too_few_subjects_raises_rather_than_leaking(self):
        """With 2 subjects a disjoint 3-way split is impossible — must fail loudly.

        Returning an empty test split would mean metrics computed on nothing.
        """
        samples = build_samples(2)
        with pytest.raises(ValueError, match="at least 3"):
            subject_disjoint_split(samples)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            subject_disjoint_split([])

    def test_bad_by_argument_raises(self):
        with pytest.raises(ValueError, match="subject.*session"):
            subject_disjoint_split(build_samples(5), by="filename")

    def test_all_splits_nonempty_with_enough_subjects(self):
        for n in (3, 5, 8, 20):
            train, val, test = subject_disjoint_split(build_samples(n), seed=0)
            assert train and val and test, f"empty split with {n} subjects"


class TestSplitIntegrity:
    def test_passes_on_a_clean_split(self):
        train, val, test = subject_disjoint_split(build_samples(9), seed=0)
        check_split_integrity(train, val, test)  # must not raise

    def test_detects_an_injected_leak(self):
        """The guard has to actually catch a leak, not just exist."""
        samples = build_samples(9)
        train, val, test = subject_disjoint_split(samples, seed=0)
        # Deliberately leak one test subject's samples into train.
        leaked_subject = next(iter({s.subject for s in test}))
        train = train + [s for s in test if s.subject == leaked_subject]

        with pytest.raises(ValueError, match="LEAKAGE"):
            check_split_integrity(train, val, test)

    @staticmethod
    def _mixed(prefix: str, n: int = 6) -> list[Sample]:
        """A both-classes-present group with subjects unique to `prefix`.

        Subjects must not repeat across the three groups, or the leak check fires
        before the assertion under test.
        """
        return [
            Sample(
                path=f"{prefix}{i}.png",
                attack_type=BF if i % 2 else PR,
                subject=f"{prefix}subj{i}",
            )
            for i in range(n)
        ]

    def test_rejects_single_class_split(self):
        """A split with no bona-fide samples makes PAD metrics undefined."""
        attacks_only = [
            Sample(path=f"a{i}.png", attack_type=PR, subject=f"atk{i}") for i in range(6)
        ]
        with pytest.raises(ValueError, match="only attack"):
            check_split_integrity(attacks_only, self._mixed("v"), self._mixed("t"))

    def test_rejects_empty_split(self):
        with pytest.raises(ValueError, match="empty"):
            check_split_integrity(self._mixed("tr"), [], self._mixed("te"))


class TestManifest:
    def test_roundtrip(self, tmp_path):
        samples = build_samples(3)
        m = DatasetManifest(samples=samples, root=str(tmp_path))
        path = tmp_path / "manifest.json"
        m.save(path)

        loaded = DatasetManifest.load(path)
        assert len(loaded.samples) == len(samples)
        assert loaded.samples[0].attack_type == samples[0].attack_type
        assert loaded.samples[0].subject == samples[0].subject
        assert loaded.root == str(tmp_path)

    def test_counts_and_describe(self):
        m = DatasetManifest(samples=build_samples(4, per_class=6))
        counts = m.counts()
        assert counts["bona_fide"] == 24
        assert counts["print"] == 24
        assert counts["replay"] == 24
        assert len(m.subjects()) == 4
        text = m.describe()
        # 4 subjects x 3 classes x 6 per class
        assert "72 samples" in text
        assert "4 subjects" in text

    def test_preserves_instrument_field(self, tmp_path):
        """Instrument is what lets eval say WHICH device defeats the model."""
        samples = [
            Sample(path="a.png", attack_type=RP, subject="s1", instrument="iphone13"),
            Sample(path="b.png", attack_type=BF, subject="s1"),
        ]
        p = tmp_path / "m.json"
        DatasetManifest(samples=samples).save(p)
        loaded = DatasetManifest.load(p)
        assert loaded.samples[0].instrument == "iphone13"
        assert loaded.samples[1].instrument is None


class TestSampleLabels:
    def test_bona_fide_is_zero_attacks_are_one(self):
        assert Sample(path="a", attack_type=BF, subject="s").label == 0
        for atk in (PR, RP, AttackType.CUTOUT, AttackType.MASK_3D):
            assert Sample(path="a", attack_type=atk, subject="s").label == 1

    def test_attack_type_is_attack_property(self):
        assert not BF.is_attack
        assert PR.is_attack
        assert RP.is_attack


class TestClassWeights:
    def test_balanced_data_gives_equal_weights(self):
        samples = [
            Sample(path=f"{i}", attack_type=BF if i % 2 == 0 else PR, subject="s")
            for i in range(20)
        ]
        w_bona, w_attack = class_weights(samples)
        assert w_bona == pytest.approx(1.0)
        assert w_attack == pytest.approx(1.0)

    def test_attack_heavy_data_upweights_bona_fide(self):
        """The typical PAD case: 1 genuine to 3 attacks.

        Without this correction the model drifts toward predicting "attack",
        which looks fine on accuracy and ruins BPCER.
        """
        samples = [Sample(path=f"b{i}", attack_type=BF, subject="s") for i in range(10)]
        samples += [Sample(path=f"a{i}", attack_type=PR, subject="s") for i in range(30)]
        w_bona, w_attack = class_weights(samples)
        assert w_bona > w_attack
        assert w_bona == pytest.approx(40 / 20)
        assert w_attack == pytest.approx(40 / 60)

    def test_single_class_falls_back_to_unweighted(self):
        samples = [Sample(path=f"{i}", attack_type=PR, subject="s") for i in range(5)]
        assert class_weights(samples) == (1.0, 1.0)
