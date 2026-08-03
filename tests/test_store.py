"""Tests for the enrollment store.

Focused on the three bugs the rewrite was meant to fix — silent data loss,
unbounded growth, non-atomic writes — plus the matching semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.contracts import EMBEDDING_DIM
from trainyourface.core.embed import l2_normalize
from trainyourface.core.store import (
    MAX_EMBEDDINGS_PER_IDENTITY,
    EnrollmentStore,
)


def make_embedding(seed: int) -> np.ndarray:
    """A deterministic unit-norm embedding."""
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=EMBEDDING_DIM).astype(np.float32))


def make_cluster(seed: int, n: int, spread: float = 0.35) -> np.ndarray:
    """n embeddings near a common center — stands in for one person's samples."""
    rng = np.random.default_rng(seed)
    center = rng.normal(size=EMBEDDING_DIM)
    noise = rng.normal(size=(n, EMBEDDING_DIM)) * spread
    return l2_normalize(center[None, :] + noise, axis=1).astype(np.float32)


@pytest.fixture
def store(tmp_path):
    return EnrollmentStore(tmp_path / "enroll.npz")


class TestBasics:
    def test_starts_empty(self, store):
        assert store.is_empty
        assert len(store) == 0
        assert store.identities == []

    def test_add_and_identify(self, store):
        emb = make_cluster(1, 5)
        kept = store.add("KAITLYN", emb, relationship="self")
        assert kept > 0
        assert "KAITLYN" in store.identities

        # A held-out sample from the same cluster should match.
        probe = make_cluster(1, 6)[5]
        ident = store.identify(probe)
        assert ident.name == "KAITLYN"
        assert ident.similarity > ident.threshold

    def test_empty_store_flags_database_empty(self, store):
        """Must distinguish "no data at all" from "no match".

        The old code returned bare None for both, so the UI could not tell the
        user whether to enroll someone or take more samples.
        """
        ident = store.identify(make_embedding(9))
        assert ident.name is None
        assert ident.database_empty is True

    def test_non_match_reports_score_not_just_none(self, store):
        store.add("A", make_cluster(2, 4))
        # A random far-away probe should not match, but should still report how
        # close it got.
        ident = store.identify(make_embedding(999), threshold=0.99)
        assert ident.name is None
        assert ident.database_empty is False
        assert ident.similarity < 0.99

    def test_empty_name_rejected(self, store):
        with pytest.raises(ValueError, match="cannot be empty"):
            store.add("   ", make_cluster(3, 2))

    def test_wrong_dimension_rejected(self, store):
        with pytest.raises(ValueError, match="expected 512-D"):
            store.add("X", np.zeros((2, 128), dtype=np.float32))


class TestRegressionSilentDataLoss:
    """The old register_face() wiped existing embeddings on re-registration."""

    def test_reenrolling_is_additive_not_destructive(self, store):
        first = make_cluster(10, 3, spread=0.5)
        store.add("SAM", first)
        count_after_first = store.count_for("SAM")
        assert count_after_first > 0

        # Re-enrolling the same name must ADD, never reset to zero.
        second = make_cluster(11, 3, spread=0.5)
        store.add("SAM", second)
        assert store.count_for("SAM") > count_after_first

    def test_adding_one_identity_does_not_disturb_another(self, store):
        store.add("A", make_cluster(20, 3, spread=0.5))
        n_a = store.count_for("A")
        store.add("B", make_cluster(21, 3, spread=0.5))
        assert store.count_for("A") == n_a
        assert store.count_for("B") > 0


class TestRegressionUnboundedGrowth:
    """The old train_face_images() appended forever with no cap or dedup."""

    def test_near_duplicates_are_dropped(self, store):
        base = make_embedding(30)
        # Ten copies of essentially the same frame.
        dupes = l2_normalize(np.tile(base, (10, 1)) + 1e-6, axis=1)
        kept = store.add("DUPE", dupes)
        assert kept == 1, f"expected 1 kept after dedup, got {kept}"

    def test_per_identity_cap_enforced(self, store):
        # Far more distinct samples than the cap allows.
        many = make_cluster(31, MAX_EMBEDDINGS_PER_IDENTITY * 3, spread=1.0)
        store.add("MANY", many)
        assert store.count_for("MANY") <= MAX_EMBEDDINGS_PER_IDENTITY

    def test_cap_holds_across_separate_calls(self, store):
        for s in range(6):
            store.add("MANY", make_cluster(40 + s, 15, spread=1.0))
        assert store.count_for("MANY") <= MAX_EMBEDDINGS_PER_IDENTITY


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "e.npz"
        s1 = EnrollmentStore(path)
        s1.add("KAITLYN", make_cluster(50, 4), relationship="self")
        s1.match_threshold = 0.42
        s1.save()

        s2 = EnrollmentStore(path)
        assert s2.identities == ["KAITLYN"]
        assert s2.count_for("KAITLYN") == s1.count_for("KAITLYN")
        assert s2.match_threshold == pytest.approx(0.42)
        assert s2.summary()[0]["relationship"] == "self"

    def test_save_is_atomic_no_tmp_left_behind(self, tmp_path):
        path = tmp_path / "e.npz"
        s = EnrollmentStore(path)
        s.add("A", make_cluster(51, 2))
        s.save()
        assert path.exists()
        assert not path.with_suffix(".npz.tmp").exists()

    def test_corrupt_store_is_survivable(self, tmp_path):
        """A corrupt file must not crash startup, and must not be deleted."""
        path = tmp_path / "bad.npz"
        path.write_bytes(b"this is not an npz file")
        s = EnrollmentStore(path)  # should warn, not raise
        assert s.is_empty
        # The bad file is preserved rather than silently destroyed.
        assert path.exists()

    def test_dimension_mismatch_store_ignored(self, tmp_path):
        """A store built with a different model must be rejected, not misused."""
        path = tmp_path / "old.npz"
        np.savez_compressed(
            path,
            embeddings=np.zeros((3, 128), dtype=np.float32),  # dlib-era 128-D
            names=np.array(["A", "A", "B"], dtype="U"),
            meta=np.array("{}"),
        )
        s = EnrollmentStore(path)
        assert s.is_empty


class TestMatching:
    def test_two_identities_separate_correctly(self, store):
        a = make_cluster(60, 8, spread=0.25)
        b = make_cluster(61, 8, spread=0.25)
        store.add("ALICE", a[:6])
        store.add("BOB", b[:6])

        # Held-out samples must go to the right identity.
        for probe in a[6:]:
            assert store.identify(probe, threshold=0.0).name == "ALICE"
        for probe in b[6:]:
            assert store.identify(probe, threshold=0.0).name == "BOB"

    def test_batch_matches_single(self, store):
        store.add("A", make_cluster(70, 5))
        store.add("B", make_cluster(71, 5))
        probes = np.stack([make_cluster(70, 6)[5], make_cluster(71, 6)[5]])

        batch = store.identify_batch(probes, threshold=0.0)
        singles = [store.identify(p, threshold=0.0) for p in probes]
        assert [i.name for i in batch] == [i.name for i in singles]
        for bi, si in zip(batch, singles, strict=True):
            assert bi.similarity == pytest.approx(si.similarity, abs=1e-5)

    def test_batch_on_empty_store(self, store):
        out = store.identify_batch(np.stack([make_embedding(1), make_embedding(2)]))
        assert len(out) == 2
        assert all(i.database_empty for i in out)

    def test_batch_empty_input(self, store):
        assert store.identify_batch(np.zeros((0, EMBEDDING_DIM), np.float32)) == []


class TestRemoval:
    def test_remove_deletes_identity_and_rows(self, store):
        store.add("A", make_cluster(80, 4))
        store.add("B", make_cluster(81, 4))
        n_before = len(store)
        removed = store.remove("A")

        assert removed > 0
        assert len(store) == n_before - removed
        assert "A" not in store.identities
        assert "B" in store.identities
        # Remaining rows must still align with remaining names.
        assert len(store._names) == len(store)

    def test_remove_everyone_leaves_valid_empty_store(self, store):
        store.add("A", make_cluster(90, 3))
        store.remove("A")
        assert store.is_empty
        assert store.identify(make_embedding(1)).database_empty


class TestCalibration:
    def test_genuine_scores_exceed_impostor(self, store):
        """Tight clusters per person -> genuine pairs should score higher."""
        store.add("A", make_cluster(100, 8, spread=0.2))
        store.add("B", make_cluster(101, 8, spread=0.2))
        genuine, impostor = store.genuine_impostor_scores()

        assert genuine.size > 0
        assert impostor.size > 0
        assert genuine.mean() > impostor.mean()

    def test_too_few_samples_returns_empty(self, store):
        store.add("A", make_cluster(110, 1))
        genuine, impostor = store.genuine_impostor_scores()
        assert genuine.size == 0
        assert impostor.size == 0
