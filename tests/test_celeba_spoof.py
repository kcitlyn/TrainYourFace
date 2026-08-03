"""Tests for the CelebA-Spoof converter and stratified reporting.

The converter is the one place in this project that reads a format it does not
control, which makes it the one place where "the file wasn't what I assumed" is a
realistic failure. So the tests lean on malformed input: truncated vectors,
unknown codes, self-contradicting labels, missing images, and paths that don't
carry an identity.

The recurring theme is that every one of those must be an error or a counted skip,
never a silent default. A converter that guesses produces a manifest with wrong
labels, and a PAD model trained on wrong labels trains perfectly happily — it just
reports numbers that mean nothing. That failure surfaces nowhere near the cause.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trainyourface.cli.main import app
from trainyourface.core.contracts import AttackType
from trainyourface.eval.report import (
    MIN_BUCKET,
    cross_dataset_report,
    evaluate_test,
    render_report,
    stratify,
)
from trainyourface.liveness.celeba_spoof import (
    SPOOF_TYPE,
    _subject_from_path,
    convert,
    parse_label,
)
from trainyourface.liveness.dataset import DatasetManifest

runner = CliRunner()


def vec(spoof_type: int, illum: int = 0, env: int = 0, live_flag: int | None = None) -> list:
    """Build a 44-element CelebA-Spoof label vector.

    `live_flag` defaults to being consistent with `spoof_type`, so a test that
    wants an inconsistent vector has to ask for it explicitly.
    """
    flag = live_flag if live_flag is not None else int(spoof_type != 0)
    return [0] * 40 + [spoof_type, illum, env, flag]


def write_celeba(tmp_path, subjects=("1", "2", "3", "4"), per_class=2, with_images=True):
    """Build a miniature CelebA-Spoof tree: real PNGs plus a label JSON.

    Mirrors the real layout — `Data/train/<identity>/{live,spoof}/<n>.png` — because
    the subject parser reads the identity out of that path, and a fixture with a
    flattened layout would test nothing.
    """
    import cv2

    root = tmp_path / "celeba"
    labels = {}
    for subj in subjects:
        for kind, code in (("live", 0), ("spoof", 9)):
            for i in range(per_class):
                rel = f"Data/train/{subj}/{kind}/{i}.png"
                # Vary illumination so stratification has more than one bucket.
                illum = 3 if int(subj) % 2 else 1
                labels[rel] = vec(code, illum=illum, env=1) if code else vec(0)
                if with_images:
                    p = root / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(p), np.full((128, 128, 3), 90 + code, np.uint8))
    meta = root / "metas" / "intra_test" / "train_label.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(labels))
    return root


class TestSubjectFromPath:
    """The identity component is what makes the split subject-disjoint.

    If this ever silently returned something per-file, `subject_disjoint_split`
    would be splitting on unique keys — a random per-image split wearing a
    subject-disjoint split's name, which is the exact leak the dataset module
    exists to prevent. So an unparseable path raises.
    """

    def test_it_finds_the_identity_directory(self):
        assert _subject_from_path("Data/train/1234/live/000001.png") == "1234"

    def test_it_ignores_numeric_filenames(self):
        """The filename is numeric too, and picking it would give one subject per image."""
        assert _subject_from_path("Data/train/77/spoof/000123.png") == "77"

    def test_a_path_without_an_identity_raises(self):
        with pytest.raises(ValueError, match="no numeric identity"):
            _subject_from_path("Data/train/live/000001.png")

    def test_the_error_explains_the_consequence(self):
        """An error here means the split can't be trusted; the message should say so."""
        with pytest.raises(ValueError, match="subject-disjoint"):
            _subject_from_path("flat.png")


class TestParseLabel:
    def test_live_gets_no_conditions(self):
        """Code 0 means "live" in all three condition fields, not a real reading.

        Recording it as illumination="live" would create a bucket that is really
        the class label, and stratifying by it would report the tautology that live
        images are classified live.
        """
        attack, cond = parse_label(vec(0))
        assert attack is AttackType.BONA_FIDE
        assert cond == {}

    @pytest.mark.parametrize(
        "code,expected",
        [
            (1, AttackType.PRINT),
            (2, AttackType.PRINT),
            (3, AttackType.PRINT),
            (7, AttackType.REPLAY),
            (8, AttackType.REPLAY),
            (9, AttackType.REPLAY),
            (4, AttackType.CUTOUT),
            (10, AttackType.MASK_3D),
        ],
    )
    def test_spoof_codes_map_by_presentation_medium(self, code, expected):
        """Paper collapses to PRINT, screens to REPLAY — what the sensor sees."""
        attack, _ = parse_label(vec(code))
        assert attack is expected

    def test_the_fine_grained_type_survives_the_collapse(self):
        """Mapping 11 classes onto 5 must not lose information."""
        _, cond = parse_label(vec(8))  # Pad -> REPLAY
        assert cond["spoof_type"] == "pad"

    def test_conditions_are_recorded(self):
        _, cond = parse_label(vec(9, illum=3, env=2))
        assert cond["illumination"] == "back"
        assert cond["environment"] == "outdoor"

    def test_every_documented_code_is_mapped(self):
        """Codes 0-10 are the full documented range; a gap would be a silent skip."""
        assert set(SPOOF_TYPE) == set(range(11))

    def test_a_truncated_vector_raises(self):
        with pytest.raises(ValueError, match="44-element"):
            parse_label([0] * 20)

    def test_an_unknown_code_raises(self):
        with pytest.raises(ValueError, match="unknown spoof-type code 99"):
            parse_label(vec(99))

    def test_a_self_contradicting_vector_raises(self):
        """The two fields encode the same fact; disagreement means format drift.

        Trusting either one alone would mislabel data silently, so neither is
        picked as authoritative.
        """
        with pytest.raises(ValueError, match="disagrees with itself"):
            parse_label(vec(9, live_flag=0))  # phone spoof labeled live

    def test_a_live_type_flagged_spoof_also_raises(self):
        with pytest.raises(ValueError, match="disagrees with itself"):
            parse_label(vec(0, live_flag=1))


class TestConvert:
    def test_it_builds_a_splittable_manifest(self, tmp_path):
        root = write_celeba(tmp_path)
        m = convert(root, log=lambda *a: None)
        assert len(m.samples) == 16  # 4 subjects x 2 classes x 2 images
        assert len(m.subjects()) == 4
        assert {s.attack_type for s in m.samples} == {
            AttackType.BONA_FIDE,
            AttackType.REPLAY,
        }

    def test_the_manifest_survives_a_save_load_round_trip(self, tmp_path):
        """`conditions` is a new field — an older loader would drop it silently."""
        root = write_celeba(tmp_path)
        m = convert(root, log=lambda *a: None)
        m.save(tmp_path / "manifest.json")
        again = DatasetManifest.load(tmp_path / "manifest.json")
        spoofs = [s for s in again.samples if s.attack_type.is_attack]
        assert spoofs and all(s.conditions.get("spoof_type") == "phone" for s in spoofs)

    def test_limit_cuts_whole_subjects_not_random_images(self, tmp_path):
        """A per-image limit would leave partial subjects, which is fine for
        splitting but wrecks the class balance within a subject. More importantly,
        the manifest must stay subject-disjoint-splittable after limiting.
        """
        root = write_celeba(tmp_path, subjects=tuple(str(i) for i in range(1, 9)))
        m = convert(root, limit=8, log=lambda *a: None)
        counts = {}
        for s in m.samples:
            counts.setdefault(s.subject, 0)
            counts[s.subject] += 1
        # Every retained subject keeps its full complement of 4 images.
        assert set(counts.values()) == {4}
        assert len(m.samples) >= 8

    def test_a_missing_root_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no such directory"):
            convert(tmp_path / "nope", log=lambda *a: None)

    def test_a_missing_label_file_names_the_flag(self, tmp_path):
        (tmp_path / "celeba").mkdir()
        with pytest.raises(FileNotFoundError, match="--label-file"):
            convert(tmp_path / "celeba", log=lambda *a: None)

    def test_images_named_but_absent_are_skipped_and_counted(self, tmp_path):
        """Silently dropping them would make the manifest size a mystery."""
        root = write_celeba(tmp_path, with_images=False)
        logged = []
        with pytest.raises(ValueError, match="none of the images it names exist"):
            convert(root, log=logged.append)

    def test_a_partially_present_dataset_reports_what_it_skipped(self, tmp_path):
        root = write_celeba(tmp_path)
        # Delete one subject's images but leave them in the annotation file.
        for p in (root / "Data" / "train" / "1").rglob("*.png"):
            p.unlink()
        logged = []
        m = convert(root, log=lambda s: logged.append(str(s)))
        assert len(m.subjects()) == 3
        assert any("not present on disk" in line for line in logged)

    def test_unparseable_labels_are_skipped_and_counted(self, tmp_path):
        root = write_celeba(tmp_path)
        meta = root / "metas" / "intra_test" / "train_label.json"
        labels = json.loads(meta.read_text())
        # Corrupt one entry into a self-contradiction.
        key = next(k for k in labels if "/spoof/" in k)
        labels[key] = vec(9, live_flag=0)
        meta.write_text(json.dumps(labels))

        logged = []
        m = convert(root, log=lambda s: logged.append(str(s)))
        assert len(m.samples) == 15
        assert any("unparseable label" in line for line in logged)

    def test_an_empty_label_file_is_an_error(self, tmp_path):
        root = tmp_path / "celeba"
        meta = root / "metas" / "intra_test" / "train_label.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{}")
        with pytest.raises(ValueError, match="non-empty JSON object"):
            convert(root, log=lambda *a: None)

    def test_all_paths_unparseable_is_an_error_not_an_empty_manifest(self, tmp_path):
        """Returning an empty manifest here would look like an empty dataset."""
        root = tmp_path / "celeba"
        meta = root / "metas" / "intra_test" / "train_label.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({"flat.png": vec(0), "other.png": vec(9)}))
        with pytest.raises(ValueError, match="no usable identity directories"):
            convert(root, log=lambda *a: None)


class TestImportCelebaCommand:
    def test_it_writes_a_manifest_and_cites_the_paper(self, tmp_path):
        root = write_celeba(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, ["import-celeba", str(root), "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "manifest.json").exists()
        assert "ECCV 2020" in result.output, "the dataset's terms require citation"

    def test_the_manifest_root_points_at_the_celeba_tree(self, tmp_path):
        """Images live in the CelebA download, not beside the manifest. If root were
        the output dir, every later image read would fail with a missing file.
        """
        root = write_celeba(tmp_path)
        out = tmp_path / "out"
        runner.invoke(app, ["import-celeba", str(root), "--out", str(out)])
        m = DatasetManifest.load(out / "manifest.json")
        assert (Path(m.root) / m.samples[0].path).exists()

    def test_a_bad_directory_is_a_message_not_a_traceback(self, tmp_path):
        result = runner.invoke(app, ["import-celeba", str(tmp_path / "gone")])
        assert result.exit_code == 1
        assert "import failed" in result.output
        assert not isinstance(result.exception, FileNotFoundError)


class TestStratify:
    """Per-condition APCER — the breakdown that says where the model fails."""

    @staticmethod
    def _samples(n_per_bucket, back_missed, normal_missed, threshold=0.5):
        """Scores where `back` misses `back_missed` attacks and `normal` misses fewer."""
        scores, types, conds = [], [], []
        for illum, missed in (("back", back_missed), ("normal", normal_missed)):
            for i in range(n_per_bucket):
                # score < threshold == wrongly accepted == missed attack
                scores.append(0.1 if i < missed else 0.9)
                types.append(AttackType.REPLAY)
                conds.append({"illumination": illum})
        return np.array(scores), types, conds

    def test_it_surfaces_a_weak_condition_the_aggregate_hides(self):
        """The whole point: 50% in backlit, 0% in normal, 25% overall.

        A reader who sees only the aggregate concludes the model is decent. A
        reader who sees the breakdown knows to avoid deploying it near a window.
        """
        scores, types, conds = self._samples(
            MIN_BUCKET, back_missed=MIN_BUCKET // 2, normal_missed=0
        )
        out = stratify(scores, types, conds, threshold=0.5)
        assert out["illumination"]["back"]["apcer"] == pytest.approx(0.5)
        assert out["illumination"]["normal"]["apcer"] == pytest.approx(0.0)

    def test_small_buckets_get_no_rate(self):
        """2-of-3 is "67%", which reads as a finding and is noise."""
        scores, types, conds = self._samples(3, back_missed=2, normal_missed=0)
        out = stratify(scores, types, conds, threshold=0.5)
        assert out["illumination"]["back"]["apcer"] is None
        assert out["illumination"]["back"]["n"] == 3
        assert out["illumination"]["back"]["missed"] == 2, "the raw count is still reported"

    def test_bona_fide_samples_are_not_bucketed(self):
        """Only APCER is stratified — see the docstring on why not BPCER."""
        out = stratify(
            np.array([0.1, 0.9]),
            [AttackType.BONA_FIDE, AttackType.REPLAY],
            [{"illumination": "back"}, {"illumination": "back"}],
            threshold=0.5,
        )
        assert out["illumination"]["back"]["n"] == 1

    def test_misaligned_inputs_raise(self):
        with pytest.raises(ValueError, match="misaligned"):
            stratify(np.array([0.1, 0.2]), [AttackType.REPLAY], [{}], threshold=0.5)

    def test_empty_conditions_produce_no_strata(self):
        out = stratify(np.array([0.9]), [AttackType.REPLAY], [{}], threshold=0.5)
        assert out == {}

    def test_it_reaches_the_report_and_renders_worst_first(self):
        scores, types, conds = self._samples(
            MIN_BUCKET, back_missed=MIN_BUCKET // 2, normal_missed=0
        )
        # Add bona-fide samples so BPCER/EER are computable.
        scores = np.concatenate([scores, np.full(MIN_BUCKET, 0.05)])
        types = types + [AttackType.BONA_FIDE] * MIN_BUCKET
        conds = conds + [{}] * MIN_BUCKET

        result = evaluate_test(scores, types, 0.5, conditions=conds)
        assert "by_condition" in result

        text = render_report(result)
        assert "APCER by capture condition" in text
        # The weak bucket must appear above the strong one.
        assert text.index("back") < text.index("normal")

    def test_no_conditions_means_no_section(self):
        """The report must not grow an empty header for a webcam dataset."""
        result = evaluate_test(
            np.array([0.9] * 5 + [0.1] * 5),
            [AttackType.REPLAY] * 5 + [AttackType.BONA_FIDE] * 5,
            0.5,
        )
        assert "by_condition" not in result
        assert "capture condition" not in render_report(result)


class TestManifestImageRoot:
    """A manifest's images don't always live beside the manifest.

    `tyf capture` writes both to the same directory, so consumers assumed --data
    was the image root. `tyf import-celeba` breaks that assumption: the manifest
    lands in your data dir while 625K images stay in the CelebA download. Training
    on an imported manifest died on the first image read with a path that didn't
    exist — while the correct root sat unread in the manifest itself.
    """

    def test_the_recorded_root_wins(self, tmp_path):
        images = tmp_path / "elsewhere"
        images.mkdir()
        m = DatasetManifest(samples=[], root=str(images))
        assert m.image_root(tmp_path / "data") == images

    def test_a_stale_root_falls_back(self, tmp_path):
        """A dataset moved wholesale should still work."""
        m = DatasetManifest(samples=[], root=str(tmp_path / "gone"))
        assert m.image_root(tmp_path) == tmp_path

    def test_an_empty_root_falls_back(self, tmp_path):
        """Manifests written before `root` was populated."""
        assert DatasetManifest(samples=[], root="").image_root(tmp_path) == tmp_path

    def test_an_imported_manifest_resolves_its_images(self, tmp_path):
        """The end-to-end version: import, then resolve every path off image_root."""
        root = write_celeba(tmp_path)
        out = tmp_path / "data"
        runner.invoke(app, ["import-celeba", str(root), "--out", str(out)])
        m = DatasetManifest.load(out / "manifest.json")
        image_root = m.image_root(out)
        assert all((image_root / s.path).exists() for s in m.samples)

    def test_the_cli_actually_uses_it(self, tmp_path):
        """Asserted through a command, not through the method.

        The method having correct behaviour is worth nothing if the callers still
        pass --data. Testing `image_root()` in isolation passes either way — this
        test is what fails when a call site regresses, and `tyf manifest --check`
        is the cheapest command that resolves every image path.
        """
        root = write_celeba(tmp_path)
        out = tmp_path / "data"
        runner.invoke(app, ["import-celeba", str(root), "--out", str(out)])

        result = runner.invoke(app, ["manifest", "--data", str(out)])
        assert result.exit_code == 0, result.output
        assert "all 16 images present" in result.output
        assert "missing" not in result.output


class TestCrossDatasetVerdict:
    """The verdict must read BOTH failure modes.

    Discrimination collapse shows in EER/AUC. Calibration collapse shows in
    APCER/BPCER while EER stays flat — the model still ranks correctly, but the
    carried-over threshold lands in the wrong place.

    The first version of `cross_dataset_report` keyed only on EER and printed "the
    gap is small, which is the good outcome" for a model whose worst-case APCER
    went 0% -> 100%. Every attack accepted, and the summary called it good.
    """

    @staticmethod
    def _result(eer, worst_apcer, bpcer=0.0, auc=1.0):
        return {"eer": eer, "worst_apcer": worst_apcer, "bpcer": bpcer, "acer": 0.0, "auc": auc}

    def test_a_threshold_that_does_not_transfer_is_called_out(self):
        """The exact case that fooled the first implementation."""
        text = cross_dataset_report(
            self._result(0.0, 0.0), self._result(0.0, 1.0), "celeba", "webcam"
        )
        assert "threshold did not transfer" in text
        assert "calibration failure" in text
        assert "good outcome" not in text, "100% APCER is not a good outcome"

    def test_it_warns_against_refitting_on_the_target_test_split(self):
        """The obvious fix for a bad threshold is also threshold-fitting on test."""
        text = cross_dataset_report(
            self._result(0.0, 0.0), self._result(0.0, 1.0), "celeba", "webcam"
        )
        assert "Do NOT re-derive it on the" in text and "TEST split" in text

    def test_discrimination_collapse_is_reported_differently(self):
        """A model that can't separate at any threshold is a different problem."""
        text = cross_dataset_report(
            self._result(0.01, 0.0), self._result(0.40, 0.5), "celeba", "webcam"
        )
        assert "discrimination collapsed" in text
        assert "ANY threshold" in text

    def test_a_genuinely_small_gap_is_called_good(self):
        text = cross_dataset_report(
            self._result(0.02, 0.03), self._result(0.025, 0.04), "celeba", "webcam"
        )
        assert "good outcome" in text

    def test_a_bpcer_only_collapse_also_trips_the_warning(self):
        """Locking every real user out is as much a failure as accepting every attack."""
        text = cross_dataset_report(
            self._result(0.0, 0.0), self._result(0.0, 0.0, bpcer=0.9), "celeba", "webcam"
        )
        assert "threshold did not transfer" in text

    def test_a_falling_auc_is_labeled_worse_not_better(self):
        """AUC improves upward while error rates improve downward.

        Signing the delta per metric is easy to get backwards, and getting it
        backwards reports a degradation as an improvement.
        """
        text = cross_dataset_report(
            self._result(0.0, 0.0, auc=0.99), self._result(0.0, 0.0, auc=0.60), "a", "b"
        )
        auc_line = next(ln for ln in text.splitlines() if "AUC" in ln)
        assert "worse" in auc_line
