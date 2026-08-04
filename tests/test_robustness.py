"""Regression tests for bugs found by deliberately abusing the code.

Every test in this file corresponds to a defect that existed and was fixed. They
are grouped by how the bug would have surfaced, because that's what makes each one
worth keeping:

SILENTLY WRONG (the dangerous kind)
    `crop_box` returned an all-black image for any face touching the left or top
    edge of the frame. The liveness model then scored a black square instead of a
    face and nothing errored — a wrong verdict presented as a real one.

CRASHES MID-DEMO (the embarrassing kind)
    A NaN embedding stored at enrollment made every later similarity NaN, which
    violated `Identity.similarity`'s [-1, 1] bound and raised a pydantic error one
    frame per face later — far from the enrollment that caused it.

FLATTERING WRONG NUMBERS (the kind this project is about)
    Raw logits passed to the PAD metrics produced a perfect 0.00% APCER. Invalid
    split fractions quietly shrank the test set to one subject.

The inputs here are not exotic. Faces at the frame edge, 4-channel frames from a
capture backend, a `--val-fraction 20` typo meaning 20% — these are things that
happen, which is why each one is pinned rather than fixed and forgotten.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from trainyourface.core.contracts import (
    EMBEDDING_DIM,
    AttackType,
    Box,
    LivenessResult,
)
from trainyourface.core.store import EnrollmentStore
from trainyourface.liveness.dataset import Sample

pytest.importorskip("cv2")


def box(x1=40, y1=40, x2=160, y2=180, score=0.99) -> Box:
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, score=score)


def emb(n=1, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, EMBEDDING_DIM)).astype(np.float32)


@pytest.fixture
def scene() -> np.ndarray:
    """A frame whose halves differ, so a lost crop is distinguishable from a dark one."""
    img = np.full((240, 320, 3), 200, np.uint8)
    img[:, :160] = 50
    return img


# ---------------------------------------------------------------------------
# silently wrong
# ---------------------------------------------------------------------------


class TestCropBoxOffFrame:
    """`image[y1:y2, x1:x2]` with a negative x1 is a NEGATIVE INDEX.

    NumPy reads from the far side of the array, the slice comes back empty, and
    `crop_box` returned its all-black fallback. So every face touching the left or
    top edge was scored as a black square by the liveness model, with no error.
    The margin path clamped already; the margin-free path (recognition's fallback
    when keypoints are missing) did not.
    """

    @pytest.mark.parametrize(
        "b",
        [
            box(-40, 50, 60, 150),  # off the left
            box(50, -40, 150, 60),  # off the top
            box(-40, -40, 60, 60),  # off the corner
            box(-300, 50, 20, 150),  # mostly off the left
        ],
    )
    @pytest.mark.parametrize("margin", [0.0, 0.3])
    def test_partially_offscreen_box_keeps_the_visible_pixels(self, scene, b, margin):
        from trainyourface.core.align import crop_box

        crop = crop_box(scene, b, margin=margin)
        assert crop.shape == (112, 112, 3)
        assert not (crop == 0).all(), "returned an all-black crop instead of the visible face"

    def test_fully_offscreen_box_still_returns_the_right_shape(self, scene):
        """Entirely outside the frame there is nothing to recover — but the caller
        stacks these into a batch, so the shape contract must hold."""
        from trainyourface.core.align import crop_box

        crop = crop_box(scene, box(-500, -500, -400, -400), size=64)
        assert crop.shape == (64, 64, 3)

    def test_crop_matches_a_manually_clamped_slice(self, scene):
        """The recovered pixels are the correct ones, not merely non-black."""
        import cv2

        from trainyourface.core.align import crop_box

        b = box(-40, 50, 60, 150)
        got = crop_box(scene, b)
        want = cv2.resize(scene[50:150, 0:60], (112, 112), interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(got, want)

    def test_offscreen_and_onscreen_crops_differ(self, scene):
        """Guards against a 'fix' that clamps everything to the same region."""
        from trainyourface.core.align import crop_box

        assert not np.array_equal(
            crop_box(scene, box(-40, 50, 60, 150)), crop_box(scene, box(200, 50, 300, 150))
        )


class TestLivenessVerdictAlignment:
    """Verdicts are matched to faces BY POSITION.

    A backend returning a different count than boxes used to raise an IndexError
    from a list comprehension. Worse than the crash is the near-miss: had the
    indexing been laxer, face 0's spoof verdict would have been attributed to
    face 1 — a spoof reported as the live person standing next to it.
    """

    def _pipe(self, n_results, n_boxes=2):
        from trainyourface.core.pipeline import FacePipeline

        boxes = [box(10, 10, 110, 110), box(150, 20, 250, 120)][:n_boxes]
        kps = [np.zeros((5, 2), np.float32)] * n_boxes

        class Detector:
            spec = type("S", (), {"name": "fake"})()
            active_provider = "Fake"

            def detect(self, image):
                return list(boxes), list(kps)

        class Embedder:
            spec = type("S", (), {"name": "fake"})()
            active_provider = "Fake"

            def embed(self, crops):
                v = np.zeros((len(crops), EMBEDDING_DIM), np.float32)
                v[:, 0] = 1.0
                return v

        class Liveness:
            path = type("P", (), {"name": "f.onnx"})()
            threshold = 0.5
            active_provider = "Fake"

            def check(self, frame, bs):
                return [LivenessResult(spoof_probability=0.01, threshold=0.5)] * n_results

        return FacePipeline(detector=Detector(), embedder=Embedder(), liveness=Liveness())

    @pytest.mark.parametrize("n_results", [0, 1, 3, 5])
    def test_count_mismatch_raises_rather_than_misattributing(self, n_results):
        frame = np.zeros((240, 320, 3), np.uint8)
        with pytest.raises(RuntimeError, match="results for"):
            self._pipe(n_results).process(frame)

    def test_matching_count_is_accepted(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        result = self._pipe(2).process(frame)
        assert len(result.faces) == 2
        assert all(f.liveness is not None for f in result.faces)


# ---------------------------------------------------------------------------
# crashes mid-demo
# ---------------------------------------------------------------------------


class TestNonFiniteEmbeddings:
    """A NaN embedding poisoned the store and crashed the viewer later.

    `identify` builds `Identity(similarity=...)`, which pydantic bounds to
    [-1, 1]; NaN fails that bound. The raise happened once per face per frame in
    the live viewer, arbitrarily long after the enrollment that caused it.
    """

    def test_nan_embeddings_are_refused_at_enrollment(self, tmp_path):
        store = EnrollmentStore(tmp_path / "s.npz")
        with pytest.raises(ValueError, match="NaN or inf"):
            store.add("ghost", np.full((1, EMBEDDING_DIM), np.nan, np.float32))
        assert store.is_empty, "a rejected add must not partially mutate the store"

    def test_inf_embeddings_are_refused(self, tmp_path):
        store = EnrollmentStore(tmp_path / "s.npz")
        with pytest.raises(ValueError, match="NaN or inf"):
            store.add("ghost", np.full((1, EMBEDDING_DIM), np.inf, np.float32))

    def test_one_bad_row_rejects_the_whole_batch(self, tmp_path):
        """Enrollment is all-or-nothing: a silently dropped frame would make the
        'kept 7 of 8' count a lie."""
        store = EnrollmentStore(tmp_path / "s.npz")
        batch = emb(4)
        batch[2, :] = np.nan
        with pytest.raises(ValueError, match="NaN or inf"):
            store.add("k", batch)
        assert len(store) == 0

    def test_identify_fails_closed_on_a_corrupt_store(self, tmp_path):
        """Even if NaN reaches the matrix another way, matching must not crash.

        Fail closed: unknown, not a match, and not an exception.
        """
        store = EnrollmentStore(tmp_path / "s.npz")
        store.add("k", emb(2))
        store._embeddings[0, :] = np.nan

        ident = store.identify(np.ones(EMBEDDING_DIM, np.float32))
        assert ident.name is None
        assert ident.similarity == 0.0

    def test_identify_batch_fails_closed_per_probe(self, tmp_path):
        store = EnrollmentStore(tmp_path / "s.npz")
        store.add("k", emb(2))
        store._embeddings[0, :] = np.nan

        out = store.identify_batch(np.ones((3, EMBEDDING_DIM), np.float32))
        assert len(out) == 3
        assert all(i.name is None for i in out)

    def test_a_zero_embedding_is_handled_without_nan(self, tmp_path):
        """l2_normalize's eps floor means a zero vector is stored, not NaN. It
        matches nothing, which is the right outcome."""
        store = EnrollmentStore(tmp_path / "s.npz")
        store.add("zed", np.zeros((1, EMBEDDING_DIM), np.float32))
        ident = store.identify(np.ones(EMBEDDING_DIM, np.float32))
        assert np.isfinite(ident.similarity)
        assert ident.name is None


class TestNonFiniteKeypoints:
    def test_nan_keypoints_raise_a_named_error(self, scene):
        """Previously surfaced as 'LinAlgError: SVD did not converge' from inside
        umeyama_similarity, naming neither the keypoints nor the frame."""
        from trainyourface.core.align import align_face

        with pytest.raises(ValueError, match="NaN or inf"):
            align_face(scene, np.full((5, 2), np.nan, np.float32))

    def test_identical_keypoints_do_not_raise(self, scene):
        """Degenerate but finite: the scale guard handles it, output is garbage but
        the pipeline keeps running."""
        from trainyourface.core.align import align_face

        out = align_face(scene, np.tile([100.0, 100.0], (5, 1)).astype(np.float32))
        assert out.shape == (112, 112, 3)
        assert np.isfinite(out).all()


class TestFreshInstallIsQuiet:
    def test_missing_store_loads_without_a_warning(self, tmp_path, capsys):
        """`tyf list` calls load() explicitly, so a new user's first interaction
        was a warning that their enrollment store could not be read."""
        store = EnrollmentStore(tmp_path / "not-created-yet.npz")
        store.load()
        assert capsys.readouterr().out == ""
        assert store.is_empty

    def test_a_genuinely_corrupt_store_still_warns(self, tmp_path, capsys):
        """The fix must not have silenced real corruption."""
        path = tmp_path / "corrupt.npz"
        path.write_bytes(b"this is not an npz file")
        store = EnrollmentStore(path)
        store.load()
        assert "could not read" in capsys.readouterr().out
        assert store.is_empty
        assert path.exists(), "a corrupt store must not be deleted"


# ---------------------------------------------------------------------------
# flattering wrong numbers
# ---------------------------------------------------------------------------


class TestScoresMustBeProbabilities:
    """Raw logits scored a perfect 0.00% APCER.

    Every logit sits well above or well below 0.5, so at the default threshold
    each class lands entirely on one side. The result is a flawless-looking
    number from a mistake, which is the exact failure mode this project's metrics
    module exists to prevent.
    """

    def test_logits_are_rejected_by_apcer_bpcer(self):
        from trainyourface.eval.metrics import apcer_bpcer

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            apcer_bpcer([-8.0, 9.5], [AttackType.BONA_FIDE, AttackType.PRINT], 0.5)

    def test_logits_are_rejected_by_evaluate(self):
        from trainyourface.eval.metrics import evaluate

        scores = [-4.0, -3.0, 5.0, 6.0]
        types = [AttackType.BONA_FIDE] * 2 + [AttackType.PRINT] * 2
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            evaluate(scores, types)

    @pytest.mark.parametrize("bad", [1.0001, -0.0001])
    def test_just_outside_the_bound_is_rejected(self, bad):
        from trainyourface.eval.metrics import apcer_bpcer

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            apcer_bpcer([0.5, bad], [AttackType.BONA_FIDE, AttackType.PRINT], 0.5)

    @pytest.mark.parametrize("edge", [0.0, 1.0])
    def test_exact_bounds_are_valid(self, edge):
        """A confident model legitimately outputs 0.0 and 1.0."""
        from trainyourface.eval.metrics import apcer_bpcer

        op = apcer_bpcer([edge, edge], [AttackType.BONA_FIDE, AttackType.PRINT], 0.5)
        assert np.isfinite(op.bpcer)


class TestSplitFractionValidation:
    """Invalid fractions silently shrank the test set instead of raising.

    `--val-fraction 20` meaning "20%" clamped to a single val group and returned
    a plausible split, so metrics came from a one-subject test set with nothing
    on screen saying so.
    """

    @staticmethod
    def _samples(n_subjects=10):
        out = []
        for i in range(n_subjects):
            for t in (AttackType.BONA_FIDE, AttackType.PRINT):
                out.append(
                    Sample(path=f"s{i}_{t.value}.png", attack_type=t, subject=f"s{i}", session="d1")
                )
        return out

    @pytest.mark.parametrize(
        ("vf", "tf"),
        [(-0.5, 0.2), (0.2, -0.5), (1.5, 0.2), (0.2, 1.5), (20.0, 20.0), (1.0, 0.2)],
    )
    def test_out_of_range_fractions_raise(self, vf, tf):
        from trainyourface.liveness.dataset import subject_disjoint_split

        with pytest.raises(ValueError, match="must be in"):
            subject_disjoint_split(self._samples(), val_fraction=vf, test_fraction=tf)

    @pytest.mark.parametrize(("vf", "tf"), [(0.5, 0.5), (0.6, 0.6), (0.95, 0.05)])
    def test_fractions_leaving_no_training_data_raise(self, vf, tf):
        from trainyourface.liveness.dataset import subject_disjoint_split

        with pytest.raises(ValueError, match="room for training"):
            subject_disjoint_split(self._samples(), val_fraction=vf, test_fraction=tf)

    def test_a_silent_fallback_to_one_group_each_is_announced(self, capsys):
        """In-range fractions can still starve training after rounding.

        vf=0.9 + tf=0.09 passes the sum check, then rounding asks for 9 val and 1
        test of 10 subjects. The clamp to 1-and-1 is a real change to what was
        requested, so it must not be silent — a test set that shrank to one
        subject still yields confident-looking metrics.
        """
        from trainyourface.liveness.dataset import subject_disjoint_split

        _, val, test = subject_disjoint_split(
            self._samples(), val_fraction=0.9, test_fraction=0.09, seed=0
        )
        assert "warning" in capsys.readouterr().out
        assert len({s.subject for s in val}) == 1
        assert len({s.subject for s in test}) == 1

    def test_normal_fractions_print_nothing(self, capsys):
        from trainyourface.liveness.dataset import subject_disjoint_split

        subject_disjoint_split(self._samples(), val_fraction=0.2, test_fraction=0.2, seed=0)
        assert capsys.readouterr().out == ""

    def test_valid_fractions_are_unaffected(self):
        from trainyourface.liveness.dataset import subject_disjoint_split

        train, val, test = subject_disjoint_split(
            self._samples(), val_fraction=0.2, test_fraction=0.2, seed=0
        )
        assert len({s.subject for s in train}) == 6
        assert len({s.subject for s in val}) == 2
        assert len({s.subject for s in test}) == 2


class TestSplitReconstructionLeak:
    """`tyf eval` re-derives the split instead of reading it, and the flags that
    control it default to 0/"subject" no matter what training used.

    So `tyf eval` on a model trained with `--seed 7` produced a DIFFERENT
    partition, putting subjects the model trained on into the "held-out" test
    set. Nothing looked wrong: the new split is still internally disjoint, so
    `check_split_integrity` passed, and the report rendered normally with
    inflated numbers. This is the worst failure mode this project has — it
    silently breaks the exact guarantee the project is built to provide.

    The fix records which groups were held out and compares against them.
    """

    @staticmethod
    def _samples(n=8):
        out = []
        for i in range(n):
            for t in (AttackType.BONA_FIDE, AttackType.PRINT):
                out.append(
                    Sample(path=f"s{i}_{t.value}.png", attack_type=t, subject=f"s{i}", session="d1")
                )
        return out

    def _fingerprint(self, seed=0, by="subject"):
        from trainyourface.liveness.dataset import split_fingerprint, subject_disjoint_split

        parts = subject_disjoint_split(self._samples(), seed=seed, by=by)
        return split_fingerprint(*parts, by=by), parts

    def test_the_same_seed_verifies(self):
        from trainyourface.liveness.dataset import verify_split_matches

        fp, (_, _, test) = self._fingerprint(seed=0)
        verify_split_matches(fp, test, by="subject")  # must not raise

    def test_a_different_seed_is_caught(self):
        from trainyourface.liveness.dataset import subject_disjoint_split, verify_split_matches

        fp, _ = self._fingerprint(seed=0)
        _, _, other_test = subject_disjoint_split(self._samples(), seed=7)
        with pytest.raises(ValueError, match="does not match"):
            verify_split_matches(fp, other_test, by="subject")

    def test_the_error_names_the_leaked_subjects(self):
        """The message has to say a trained-on subject leaked in, not just that
        two lists differ — otherwise the natural response is to pass --seed 7 and
        make the warning go away."""
        from trainyourface.liveness.dataset import subject_disjoint_split, verify_split_matches

        fp, _ = self._fingerprint(seed=0)
        _, _, other_test = subject_disjoint_split(self._samples(), seed=7)
        with pytest.raises(ValueError, match="TRAINED on"):
            verify_split_matches(fp, other_test, by="subject")

    def test_a_different_split_mode_is_caught(self):
        from trainyourface.liveness.dataset import subject_disjoint_split, verify_split_matches

        fp, _ = self._fingerprint(seed=0, by="subject")
        _, _, sess_test = subject_disjoint_split(self._samples(), seed=0, by="session")
        with pytest.raises(ValueError, match="--split-by"):
            verify_split_matches(fp, sess_test, by="session")

    def test_a_fingerprint_records_disjoint_groups(self):
        fp, _ = self._fingerprint(seed=0)
        assert not set(fp["train"]) & set(fp["test"])
        assert not set(fp["train"]) & set(fp["val"])
        assert not set(fp["val"]) & set(fp["test"])

    def test_an_empty_fingerprint_is_skipped_not_treated_as_a_match(self):
        """Checkpoints predating this have no recorded split. 'Cannot verify' and
        'verified' are different claims, so the caller is told which it has."""
        from trainyourface.liveness.dataset import verify_split_matches

        _, (_, _, test) = self._fingerprint(seed=0)
        verify_split_matches({}, test, by="subject")  # no data, no verdict

    def test_load_split_returns_none_when_absent(self, tmp_path):
        import json

        from trainyourface.liveness.predict import load_split

        assert load_split(tmp_path) is None
        (tmp_path / "train_summary.json").write_text(json.dumps({"val_threshold": 0.5}))
        assert load_split(tmp_path) is None, "a summary without a split is not a split"

    def test_load_split_survives_a_corrupt_summary(self, tmp_path):
        from trainyourface.liveness.predict import load_split

        (tmp_path / "train_summary.json").write_text("{not json")
        assert load_split(tmp_path) is None

    def test_load_split_reads_a_recorded_split(self, tmp_path):
        import json

        from trainyourface.liveness.predict import load_split

        fp, _ = self._fingerprint(seed=0)
        (tmp_path / "train_summary.json").write_text(json.dumps({"split": fp}))
        assert load_split(tmp_path) == fp


class TestPreprocessingShapeContract:
    """A 4-channel frame produced a (4, H, W) tensor instead of raising.

    Some capture backends and PNG reads yield BGRA. The extra channel passed
    straight through the reverse-and-transpose and failed deep inside the model,
    with an error naming neither this function nor the offending image.
    """

    @pytest.mark.parametrize(
        "shape", [(64, 64, 4), (64, 64, 1), (64, 64), (3, 64, 64), (64, 64, 3, 1)]
    )
    def test_wrong_channel_count_raises_here(self, shape):
        from trainyourface.liveness.dataset import to_model_input

        with pytest.raises(ValueError, match="H, W, 3"):
            to_model_input(np.zeros(shape, np.uint8))

    def test_valid_bgr_still_works(self):
        from trainyourface.liveness.dataset import to_model_input

        out = to_model_input(np.zeros((64, 64, 3), np.uint8))
        assert out.shape == (3, 64, 64)


class TestBoxScaledDegenerate:
    """Downscaling a small box rounded both edges onto one pixel, failing Box's
    own x2 > x1 validator — a legitimate operation raising a validation error."""

    @pytest.mark.parametrize("factor", [0.5, 0.1, 0.01, 0.0])
    def test_scaling_a_tiny_box_stays_valid(self, factor):
        out = box(10, 10, 11, 11).scaled(factor, factor)
        assert out.width >= 1 and out.height >= 1

    def test_normal_scaling_is_exact(self):
        out = box(10, 20, 110, 220).scaled(2.0, 0.5)
        assert (out.x1, out.y1, out.x2, out.y2) == (20, 10, 220, 110)


class TestManifestPortability:
    """A manifest is a committable, cross-machine artifact — so it must survive
    the machine that wrote it.

    SILENTLY WRONG: `tyf capture` built paths with `str(Path("images") / label /
    name)`, which on Windows produces `images\\bona_fide\\s1_0.png`. Read back on
    macOS or Linux that is a SINGLE filename containing backslashes, not a path:
    every image read fails and the subject parser can't find the identity dir. The
    project's whole premise is cross-platform, so a manifest that only works on the
    OS that made it is a real defect.
    """

    def test_a_windows_path_is_normalized_on_construction(self):
        from trainyourface.core.contracts import AttackType
        from trainyourface.liveness.dataset import Sample

        s = Sample(
            path="images\\bona_fide\\s1_0.png", attack_type=AttackType.BONA_FIDE, subject="s"
        )
        assert s.path == "images/bona_fide/s1_0.png"

    def test_a_posix_path_is_left_alone(self):
        from trainyourface.core.contracts import AttackType
        from trainyourface.liveness.dataset import Sample

        s = Sample(path="images/bona_fide/s1_0.png", attack_type=AttackType.BONA_FIDE, subject="s")
        assert s.path == "images/bona_fide/s1_0.png"

    def test_a_windows_manifest_round_trips_to_forward_slashes(self, tmp_path):
        from trainyourface.core.contracts import AttackType
        from trainyourface.liveness.dataset import DatasetManifest, Sample

        m = DatasetManifest(
            samples=[Sample("Data\\train\\7\\live\\1.png", AttackType.BONA_FIDE, "7")],
            root=str(tmp_path),
        )
        m.save(tmp_path / "manifest.json")
        reloaded = DatasetManifest.load(tmp_path / "manifest.json")
        assert reloaded.samples[0].path == "Data/train/7/live/1.png"

    def test_a_windows_celeba_path_still_yields_its_subject(self):
        """The subject parser splits on '/', so a normalized path is what lets it
        find the identity directory regardless of the capture OS."""
        from trainyourface.core.contracts import AttackType
        from trainyourface.liveness.celeba_spoof import _subject_from_path
        from trainyourface.liveness.dataset import Sample

        s = Sample("Data\\train\\1234\\live\\1.png", AttackType.BONA_FIDE, "x")
        assert _subject_from_path(s.path) == "1234"


class TestManifestLoadValidation:
    """A manifest gets hand-edited, script-generated, and git-merged, so "it
    parsed" is not "it is usable".

    CRASHES FAR FROM THE CAUSE: every malformed manifest used to surface as a raw
    JSONDecodeError, KeyError, or AttributeError naming a dict key or a byte
    offset. The worst was a non-dict `conditions`, which passed straight through
    load() and died later inside `fairness_audit` as `'str' object has no attribute
    'items'` — arbitrarily far from the file that caused it.
    """

    @staticmethod
    def _write(tmp_path, content):
        p = tmp_path / "manifest.json"
        p.write_text(content)
        return p

    def test_corrupt_json_names_the_file(self, tmp_path):
        from trainyourface.liveness.dataset import DatasetManifest

        with pytest.raises(ValueError, match="not valid JSON"):
            DatasetManifest.load(self._write(tmp_path, "hello{"))

    def test_a_top_level_array_is_rejected(self, tmp_path):
        from trainyourface.liveness.dataset import DatasetManifest

        with pytest.raises(ValueError, match="must contain a JSON object"):
            DatasetManifest.load(self._write(tmp_path, "[1, 2, 3]"))

    def test_samples_must_be_a_list(self, tmp_path):
        from trainyourface.liveness.dataset import DatasetManifest

        with pytest.raises(ValueError, match="'samples' must be a list"):
            DatasetManifest.load(self._write(tmp_path, '{"samples": "nope"}'))

    def test_a_missing_required_field_names_it(self, tmp_path):
        from trainyourface.liveness.dataset import DatasetManifest

        content = '{"samples":[{"attack_type":"bona_fide","subject":"s"}]}'
        with pytest.raises(ValueError, match="missing required field.*path"):
            DatasetManifest.load(self._write(tmp_path, content))

    def test_an_unknown_attack_type_lists_the_valid_ones(self, tmp_path):
        from trainyourface.liveness.dataset import DatasetManifest

        content = '{"samples":[{"path":"a.png","attack_type":"laser","subject":"s"}]}'
        with pytest.raises(ValueError, match="not one of"):
            DatasetManifest.load(self._write(tmp_path, content))

    def test_non_dict_conditions_are_caught_at_load_not_in_the_report(self, tmp_path):
        """The whole point: fail at the file, not three modules downstream."""
        from trainyourface.liveness.dataset import DatasetManifest

        content = (
            '{"samples":[{"path":"a.png","attack_type":"bona_fide","subject":"s",'
            '"conditions":"attr:Male"}]}'
        )
        with pytest.raises(ValueError, match="'conditions' of type str"):
            DatasetManifest.load(self._write(tmp_path, content))

    def test_a_null_root_normalizes_to_empty_string(self, tmp_path):
        """root=None would break image_root()'s Path(self.root). Coerced to ''."""
        from trainyourface.liveness.dataset import DatasetManifest

        m = DatasetManifest.load(self._write(tmp_path, '{"root":null,"samples":[]}'))
        assert m.root == ""
        assert m.image_root(tmp_path) == tmp_path

    def test_a_valid_manifest_still_loads(self, tmp_path):
        from trainyourface.liveness.dataset import DatasetManifest

        content = (
            '{"root":"/x","samples":[{"path":"a.png","attack_type":"bona_fide","subject":"s"}]}'
        )
        m = DatasetManifest.load(self._write(tmp_path, content))
        assert len(m.samples) == 1 and m.root == "/x"


class TestCorruptManifestThroughTheCLI:
    """A malformed manifest must be a message, not a traceback with no output.

    The commands exited 1 only because typer catches the escaping ValueError; the
    user saw a stack and no explanation. Someone hand-editing a manifest should be
    told what's wrong with it.
    """

    def _bad(self, tmp_path):
        (tmp_path / "manifest.json").write_text("hello{")
        return tmp_path

    def test_manifest_command_explains_it(self, tmp_path):
        from typer.testing import CliRunner

        from trainyourface.cli.main import app

        result = CliRunner().invoke(app, ["manifest", "--data", str(self._bad(tmp_path))])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output
        assert not isinstance(result.exception, Exception) or isinstance(
            result.exception, SystemExit
        )

    def test_train_command_explains_it(self, tmp_path):
        from typer.testing import CliRunner

        from trainyourface.cli.main import app

        result = CliRunner().invoke(
            app, ["train", "--data", str(self._bad(tmp_path)), "--epochs", "1"]
        )
        assert result.exit_code == 1
        assert "not valid JSON" in result.output


class TestCheckpointDeserialization:
    """A PyTorch checkpoint is a pickle, so loading one runs code from it.

    ARBITRARY CODE EXECUTION: all four load sites passed `weights_only=False`,
    which lets a crafted `.pt` execute anything via `__reduce__` on load. This is
    not theoretical — the test below builds such a file and confirms the payload is
    blocked. The exposure is direct: `tyf eval --checkpoint someone-elses.pt` is
    the documented way to reproduce a reported number, and PAD models are exactly
    the artifact people pass around.

    Nothing is lost by restricting it: these checkpoints hold tensors, plain dicts,
    ints and floats, all of which `weights_only=True` permits.
    """

    pytestmark = pytest.mark.train

    def test_a_malicious_checkpoint_cannot_execute_code(self, tmp_path):
        """The payload is a local file touch, standing in for a real one."""
        import os

        import torch

        from trainyourface.liveness.model import load_checkpoint

        marker = tmp_path / "EXECUTED"

        class Payload:
            def __reduce__(self):
                return (os.system, (f"touch {marker}",))

        evil = tmp_path / "evil.pt"
        torch.save({"model_state": Payload(), "model_config": {}}, evil)

        with pytest.raises(ValueError, match="could not safely load"):
            load_checkpoint(evil, map_location="cpu")
        assert not marker.exists(), "code from the checkpoint ran"

    def test_the_error_names_the_escape_hatch_without_taking_it(self, tmp_path):
        """A trusted-but-odd checkpoint should be recoverable, but only
        deliberately. Silently retrying with weights_only=False would defeat the
        whole guard, so the message explains add_safe_globals instead."""
        import os

        import torch

        from trainyourface.liveness.model import load_checkpoint

        class Payload:
            def __reduce__(self):
                return (os.system, ("true",))

        evil = tmp_path / "odd.pt"
        torch.save({"x": Payload()}, evil)
        with pytest.raises(ValueError) as exc:
            load_checkpoint(evil, map_location="cpu")
        assert "add_safe_globals" in str(exc.value)

    def test_a_legitimate_checkpoint_still_loads(self, tmp_path):
        """The guard must not break the normal path."""
        import torch

        from trainyourface.liveness.model import load_checkpoint

        good = tmp_path / "good.pt"
        torch.save(
            {
                "model_state": {"w": torch.zeros(2, 2)},
                "model_config": {"width": 16},
                "epoch": 3,
                "val_eer": 0.05,
            },
            good,
        )
        ckpt = load_checkpoint(good, map_location="cpu")
        assert ckpt["epoch"] == 3
        assert ckpt["model_config"]["width"] == 16

    def test_no_unsafe_load_remains_in_the_source(self):
        """A guard against reintroduction. `weights_only=False` is one word in a
        diff and reopens the hole completely, which is easy to miss in review.

        Parsed with `ast` rather than grepped: the docstrings that explain this
        vulnerability necessarily contain the string, and a text search flags them
        as offenders. Matching actual keyword arguments in real calls is the check
        that means something.
        """
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "src"
        offenders = []
        for path in src.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if (
                        kw.arg == "weights_only"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False
                    ):
                        offenders.append(f"{path.relative_to(src)}:{node.lineno}")
        assert not offenders, f"unsafe torch.load at {offenders}"

    def test_that_guard_would_actually_catch_a_regression(self):
        """Verifies the detector, not the codebase.

        A source-scanning test that silently matches nothing is worthless — it
        passes forever regardless of what the code does. This feeds it a known-bad
        snippet to prove the AST walk finds one.
        """
        import ast

        tree = ast.parse("torch.load(p, map_location='cpu', weights_only=False)")
        found = [
            kw
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "weights_only"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is False
        ]
        assert len(found) == 1


class TestBiometricDataAtRest:
    """Face embeddings are biometric identifiers, and unlike a password you
    cannot issue someone a new face.

    The store was written with the default 0o644, leaving templates readable by
    every local user and process. The default directory is 0o700, which masks the
    problem until someone passes `--store` somewhere else.
    """

    # POSIX only. On Windows `os.chmod` can only toggle the read-only flag — the
    # permission bits are emulated and always report group/other as readable
    # (0o666), so asserting 0o600 there tests the emulation, not this code. The
    # source comment already said this; the first version of these tests ignored it
    # and failed both Windows jobs. Confidentiality on Windows rests on the user
    # profile directory's ACL instead.
    posix_only = pytest.mark.skipif(
        os.name != "posix", reason="POSIX permission bits are emulated on Windows"
    )

    @posix_only
    def test_the_store_is_owner_only(self, tmp_path):
        import stat

        from trainyourface.core.store import EnrollmentStore

        path = tmp_path / "enrollments.npz"
        s = EnrollmentStore(path)
        s.add("kc", np.zeros((1, 512), dtype=np.float32))
        s.save()

        mode = stat.S_IMODE(path.stat().st_mode)
        assert not mode & stat.S_IROTH, "world-readable biometric templates"
        assert not mode & stat.S_IRGRP, "group-readable biometric templates"

    @posix_only
    def test_permissions_survive_a_rewrite(self, tmp_path):
        """save() replaces the file, so the mode has to be reapplied each time."""
        import stat

        from trainyourface.core.store import EnrollmentStore

        path = tmp_path / "enrollments.npz"
        s = EnrollmentStore(path)
        s.add("a", np.zeros((1, 512), dtype=np.float32))
        s.save()
        s.add("b", np.ones((1, 512), dtype=np.float32))
        s.save()
        assert not stat.S_IMODE(path.stat().st_mode) & stat.S_IROTH

    def test_the_chmod_is_attempted_on_every_platform(self, tmp_path, monkeypatch):
        """Windows can't enforce the mode, but the call must still be made.

        Skipping the two tests above on Windows would otherwise leave the chmod
        itself untested there — so a refactor that dropped it would go unnoticed on
        that platform. This asserts the attempt, which is platform-independent.
        """
        from pathlib import Path

        from trainyourface.core.store import EnrollmentStore

        seen: list[int] = []
        real_chmod = Path.chmod

        def spy(self, mode, **kwargs):
            seen.append(mode)
            return real_chmod(self, mode, **kwargs)

        monkeypatch.setattr(Path, "chmod", spy)
        s = EnrollmentStore(tmp_path / "enrollments.npz")
        s.add("kc", np.zeros((1, 512), dtype=np.float32))
        s.save()
        assert 0o600 in seen, "save() must restrict the biometric store's mode"

    def test_the_store_still_round_trips(self, tmp_path):
        from trainyourface.core.store import EnrollmentStore

        path = tmp_path / "enrollments.npz"
        s = EnrollmentStore(path)
        s.add("kc", np.zeros((1, 512), dtype=np.float32))
        s.save()
        assert EnrollmentStore(path).identities == ["kc"]

    def test_the_store_refuses_pickled_arrays(self):
        """np.load(allow_pickle=True) is the numpy equivalent of the torch hole.

        Asserted at the source level: an .npz is a zip of .npy files, and with
        pickle allowed a crafted store would execute code the same way.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "src"
        offenders = [str(p) for p in src.rglob("*.py") if "allow_pickle=True" in p.read_text()]
        assert not offenders, f"unsafe np.load at {offenders}"


class TestDownloadBounds:
    """SHA-256 verification is the real defence for downloads, and it is already
    in place for both the archive and the extracted member.

    This covers the narrower gap: bytes are streamed to disk BEFORE the hash is
    checked, so without a cap a hostile mirror or a zip bomb could fill the cache
    directory even though the result would then be rejected.
    """

    def test_an_oversized_stream_is_aborted(self):
        import io

        from trainyourface.core.models import ModelDownloadError, _copy_bounded

        with pytest.raises(ModelDownloadError, match="exceeded"):
            _copy_bounded(io.BytesIO(b"x" * 5000), io.BytesIO(), limit=1000, what="test")

    def test_a_normal_stream_passes_through_intact(self):
        import io

        from trainyourface.core.models import _copy_bounded

        out = io.BytesIO()
        _copy_bounded(io.BytesIO(b"y" * 500), out, limit=1000)
        assert out.getvalue() == b"y" * 500

    def test_the_cap_leaves_real_headroom(self):
        """The artifacts total ~17 MB. A cap near that would break on a legitimate
        model update; one in the gigabytes would not be a cap."""
        from trainyourface.core.models import MAX_ARTIFACT_BYTES

        assert 50 * 1024 * 1024 < MAX_ARTIFACT_BYTES < 1024 * 1024 * 1024

    def test_every_model_url_uses_tls(self):
        from trainyourface.core.models import DETECTOR, EMBEDDER

        for spec in (DETECTOR, EMBEDDER):
            assert spec.url.startswith("https://"), f"{spec.name} downloads over plaintext"

    def test_every_model_is_hash_pinned(self):
        """An unpinned artifact means a replaced upstream release goes unnoticed."""
        from trainyourface.core.models import DETECTOR, EMBEDDER

        for spec in (DETECTOR, EMBEDDER):
            assert spec.sha256, f"{spec.name} has no expected hash"
            if spec.archive_member:
                assert spec.archive_sha256, f"{spec.name} archive is unpinned"
