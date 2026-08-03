"""End-to-end CLI tests through typer's runner.

`src/trainyourface/cli/main.py` was the largest untested file in the repo, which
is the wrong place to have no tests: it's the only part of the project a user
actually touches, and its failure modes are the ones that make someone give up on
the tool — a confusing error on a fresh install, a command that exits 0 having
done nothing, a wrong exit code that breaks a script.

Every test here invokes a real command through `CliRunner` and asserts on the exit
code and the output text. Commands needing a camera, a display, or model weights
are exercised only up to the point where they'd need them, which is where the
argument handling and the guard rails live.

WHY EXIT CODES ARE ASSERTED EVERYWHERE
--------------------------------------
`tyf train` refusing a 2-subject dataset is only useful if it exits non-zero — a
CI pipeline or shell script checks `$?`, not prose. Two of the bugs pinned below
are commands that printed a failure and still exited 0.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from trainyourface.cli.main import app
from trainyourface.core.contracts import EMBEDDING_DIM
from trainyourface.core.store import EnrollmentStore

runner = CliRunner()


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "enrollments.npz"


def emb(n=1, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, EMBEDDING_DIM)).astype(np.float32)


def write_manifest(tmp_path, subjects=("s1", "s2", "s3"), with_images=True):
    """A minimal on-disk PAD dataset: manifest plus real 128x128 PNGs."""
    import cv2

    from trainyourface.core.contracts import AttackType
    from trainyourface.liveness.dataset import DatasetManifest, Sample

    samples = []
    for subj in subjects:
        for t in (AttackType.BONA_FIDE, AttackType.PRINT):
            for i in range(2):
                rel = f"images/{t.value}/{subj}_{i}.png"
                samples.append(
                    Sample(path=rel, attack_type=t, subject=subj, session="d1", instrument="x")
                )
                if with_images:
                    p = tmp_path / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(p), np.full((128, 128, 3), 100, np.uint8))
    m = DatasetManifest(samples=samples, root=str(tmp_path))
    m.save(tmp_path / "manifest.json")
    return m


# ---------------------------------------------------------------------------
# the entry point itself
# ---------------------------------------------------------------------------


class TestEntryPoint:
    def test_bare_invocation_shows_help_not_a_traceback(self):
        result = runner.invoke(app, [])
        assert "Usage" in result.output

    def test_help_lists_every_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in (
            "watch",
            "enroll",
            "list",
            "forget",
            "capture",
            "manifest",
            "train",
            "eval",
            "calibrate",
            "export",
            "bench",
            "info",
        ):
            assert cmd in result.output, f"{cmd} missing from --help"

    @pytest.mark.parametrize(
        "cmd",
        [
            "watch",
            "enroll",
            "list",
            "forget",
            "capture",
            "manifest",
            "train",
            "eval",
            "calibrate",
            "export",
            "bench",
            "info",
        ],
    )
    def test_every_subcommand_help_resolves(self, cmd):
        """A lazy import inside a command body means `--help` is the only cheap
        check that the module it imports still exists."""
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, result.output

    def test_unknown_command_exits_nonzero(self):
        result = runner.invoke(app, ["nonsense"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# enrollment store commands
# ---------------------------------------------------------------------------


class TestListIdentities:
    def test_fresh_install_says_nothing_is_enrolled(self, store_path):
        """No traceback, no scary warning, and exit 0 — nothing is wrong."""
        result = runner.invoke(app, ["list", "--store", str(store_path)])
        assert result.exit_code == 0
        assert "no identities enrolled" in result.output
        assert "could not read" not in result.output

    def test_lists_enrolled_identities_with_counts(self, store_path):
        s = EnrollmentStore(store_path)
        s.add("kaitlyn", emb(3, seed=1), relationship="self")
        s.add("alex", emb(2, seed=2))
        s.save()

        result = runner.invoke(app, ["list", "--store", str(store_path)])
        assert result.exit_code == 0
        assert "kaitlyn" in result.output and "alex" in result.output
        assert "self" in result.output

    def test_unicode_names_survive_the_round_trip(self, store_path):
        s = EnrollmentStore(store_path)
        s.add("日本語", emb(1, seed=3))
        s.save()
        result = runner.invoke(app, ["list", "--store", str(store_path)])
        assert result.exit_code == 0
        assert "日本語" in result.output


class TestForget:
    def test_forgetting_an_unknown_name_exits_nonzero(self, store_path):
        """A script doing `tyf forget X && echo gone` must not print 'gone'."""
        result = runner.invoke(app, ["forget", "nobody", "--store", str(store_path), "--yes"])
        assert result.exit_code == 1
        assert "not enrolled" in result.output

    def test_forget_removes_the_identity_and_persists(self, store_path):
        s = EnrollmentStore(store_path)
        s.add("kaitlyn", emb(3, seed=1))
        s.add("alex", emb(2, seed=2))
        s.save()

        result = runner.invoke(app, ["forget", "kaitlyn", "--store", str(store_path), "--yes"])
        assert result.exit_code == 0

        # Reload from disk: erasure has to be durable, not just in-memory.
        after = EnrollmentStore(store_path)
        assert "kaitlyn" not in after.identities
        assert "alex" in after.identities
        assert after.count_for("kaitlyn") == 0

    def test_forget_without_yes_aborts_and_keeps_the_data(self, store_path):
        """Biometric deletion is irreversible; declining must be a real no-op."""
        s = EnrollmentStore(store_path)
        s.add("kaitlyn", emb(3, seed=1))
        s.save()

        result = runner.invoke(app, ["forget", "kaitlyn", "--store", str(store_path)], input="n\n")
        assert result.exit_code != 0
        assert EnrollmentStore(store_path).count_for("kaitlyn") == 3


# ---------------------------------------------------------------------------
# dataset commands
# ---------------------------------------------------------------------------


class TestManifestCommand:
    def test_missing_manifest_exits_nonzero_with_a_next_step(self, tmp_path):
        result = runner.invoke(app, ["manifest", "--data", str(tmp_path)])
        assert result.exit_code == 1
        assert "tyf capture" in result.output, "an error should name the fix"

    def test_reports_counts_subjects_and_the_split(self, tmp_path):
        write_manifest(tmp_path)
        result = runner.invoke(app, ["manifest", "--data", str(tmp_path)])
        assert result.exit_code == 0
        assert "12 samples, 3 subjects" in result.output
        assert "subject-disjoint split" in result.output
        assert "all 12 images present" in result.output

    def test_missing_image_files_are_reported_and_exit_nonzero(self, tmp_path):
        """A manifest listing images that aren't there would fail at train time
        with a FileNotFoundError from inside a DataLoader worker."""
        write_manifest(tmp_path)
        (tmp_path / "images/print/s1_0.png").unlink()

        result = runner.invoke(app, ["manifest", "--data", str(tmp_path)])
        assert result.exit_code == 1
        assert "missing" in result.output

    def test_too_few_subjects_is_reported_as_not_splittable(self, tmp_path):
        """Two subjects cannot produce a disjoint train/val/test split."""
        write_manifest(tmp_path, subjects=("s1", "s2"))
        result = runner.invoke(app, ["manifest", "--data", str(tmp_path)])
        assert result.exit_code == 1
        assert "not splittable" in result.output
        assert "at least 3" in result.output

    def test_no_check_skips_file_verification(self, tmp_path):
        write_manifest(tmp_path)
        (tmp_path / "images/print/s1_0.png").unlink()
        result = runner.invoke(app, ["manifest", "--data", str(tmp_path), "--no-check"])
        assert result.exit_code == 0


class TestCaptureArgumentHandling:
    def test_an_invalid_label_is_rejected_before_the_camera_opens(self, tmp_path):
        """Labels are the dataset's ground truth. A typo'd label that fell through
        to a free-form string would mislabel every frame in the session."""
        result = runner.invoke(
            app,
            ["capture", "--label", "deepfake", "--subject", "s1", "--out", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_label_and_subject_are_both_required(self, tmp_path):
        assert runner.invoke(app, ["capture", "--subject", "s1"]).exit_code != 0
        assert runner.invoke(app, ["capture", "--label", "print"]).exit_code != 0

    def test_target_must_be_positive(self, tmp_path):
        result = runner.invoke(
            app,
            ["capture", "--label", "print", "--subject", "s1", "--target", "0"],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# train / eval guard rails
# ---------------------------------------------------------------------------


class TestTrainGuardRails:
    def test_missing_manifest_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["train", "--data", str(tmp_path), "--out", str(tmp_path / "runs")]
        )
        assert result.exit_code == 1
        assert "tyf capture" in result.output

    def test_two_subjects_is_refused_rather_than_silently_leaked(self, tmp_path):
        """The headline anti-leakage guarantee, enforced at the CLI boundary.

        Training on a split that reuses subjects across train and test is how a
        PAD project reports 99% and means nothing, so this must fail loudly
        instead of falling back to a random split.
        """
        write_manifest(tmp_path, subjects=("s1", "s2"))
        result = runner.invoke(
            app, ["train", "--data", str(tmp_path), "--out", str(tmp_path / "runs")]
        )
        assert result.exit_code == 1
        assert "split failed" in result.output

    def test_batch_size_below_two_is_rejected(self, tmp_path):
        """BatchNorm cannot compute statistics over a single sample."""
        write_manifest(tmp_path)
        result = runner.invoke(app, ["train", "--data", str(tmp_path), "--batch-size", "1"])
        assert result.exit_code != 0

    def test_zero_epochs_is_rejected(self, tmp_path):
        result = runner.invoke(app, ["train", "--data", str(tmp_path), "--epochs", "0"])
        assert result.exit_code != 0


class TestEvalGuardRails:
    """`eval` used to let two failures escape as raw tracebacks.

    It's the command most likely to be pointed at someone else's checkpoint or a
    moved dataset, and it was the only one whose guards were missing: a missing
    manifest surfaced as `FileNotFoundError` and a too-small dataset as a bare
    `ValueError`, both naming a path without saying which flag to change.
    """

    @staticmethod
    def _run_dir(tmp_path):
        run = tmp_path / "runs"
        run.mkdir(exist_ok=True)
        (run / "best.pt").write_bytes(b"placeholder")
        (run / "train_summary.json").write_text(json.dumps({"val_threshold": 0.5}))
        return run

    def test_missing_checkpoint_exits_nonzero(self, tmp_path):
        write_manifest(tmp_path)
        result = runner.invoke(
            app, ["eval", "--checkpoint", str(tmp_path / "nope.pt"), "--data", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "no checkpoint" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_missing_manifest_is_a_message_not_a_traceback(self, tmp_path):
        run = self._run_dir(tmp_path)
        result = runner.invoke(
            app, ["eval", "--checkpoint", str(run / "best.pt"), "--data", str(tmp_path / "gone")]
        )
        assert result.exit_code == 1
        assert "no manifest" in result.output
        assert "--data" in result.output, "the error should name the flag to fix"
        assert not isinstance(result.exception, FileNotFoundError)

    def test_an_unsplittable_dataset_is_a_message_not_a_traceback(self, tmp_path):
        data = tmp_path / "data"
        write_manifest(data, subjects=("s1", "s2"))
        run = self._run_dir(tmp_path)
        result = runner.invoke(
            app, ["eval", "--checkpoint", str(run / "best.pt"), "--data", str(data)]
        )
        assert result.exit_code == 1
        assert "split failed" in result.output
        assert not isinstance(result.exception, ValueError)

    def test_no_threshold_and_no_summary_is_refused(self, tmp_path):
        """Rather than defaulting to 0.5 and reporting numbers at a threshold
        nobody chose."""
        write_manifest(tmp_path)
        ckpt = tmp_path / "bare" / "best.pt"
        ckpt.parent.mkdir()
        ckpt.write_bytes(b"placeholder")

        result = runner.invoke(app, ["eval", "--checkpoint", str(ckpt), "--data", str(tmp_path)])
        assert result.exit_code == 1
        assert "--threshold" in result.output
        assert "hand-chosen" in result.output


class TestCalibrate:
    def test_calibrate_needs_at_least_two_identities(self, store_path):
        """Impostor pairs require two people; with one there is nothing to
        separate and a threshold cannot be derived."""
        s = EnrollmentStore(store_path)
        s.add("solo", emb(4, seed=1))
        s.save()

        result = runner.invoke(app, ["calibrate", "--store", str(store_path)])
        assert result.exit_code != 0
        assert "identit" in result.output.lower()

    def test_calibrate_on_an_empty_store_exits_nonzero(self, store_path):
        result = runner.invoke(app, ["calibrate", "--store", str(store_path)])
        assert result.exit_code != 0

    def test_separable_identities_get_a_threshold_inside_the_margin(self, store_path):
        """Two well-separated identities: the suggested threshold must sit between
        the impostor maximum and the genuine minimum, or it isn't a decision
        boundary at all."""
        rng = np.random.default_rng(0)
        a = np.zeros((5, EMBEDDING_DIM), np.float32)
        a[:, 0] = 1.0
        a += rng.normal(0, 0.01, a.shape).astype(np.float32)
        b = np.zeros((5, EMBEDDING_DIM), np.float32)
        b[:, 1] = 1.0
        b += rng.normal(0, 0.01, b.shape).astype(np.float32)

        s = EnrollmentStore(store_path)
        s.add("a", a)
        s.add("b", b)
        s.save()

        result = runner.invoke(app, ["calibrate", "--store", str(store_path)])
        assert result.exit_code == 0, result.output
        assert "fully separable" in result.output

        genuine, impostor = s.genuine_impostor_scores()
        suggested = float(result.output.split("suggested threshold:")[1].split()[0])
        assert impostor.max() < suggested < genuine.min()

    def test_overlapping_distributions_are_reported_as_not_error_free(self, store_path):
        """Random embeddings from two people overlap. The honest output says no
        threshold is error-free rather than presenting one as if it were clean."""
        s = EnrollmentStore(store_path)
        s.add("a", emb(5, seed=10))
        s.add("b", emb(5, seed=11))
        s.save()

        result = runner.invoke(app, ["calibrate", "--store", str(store_path)])
        assert result.exit_code == 0, result.output
        assert "overlap" in result.output
        assert "no threshold is error-free" in result.output

    def test_calibrate_always_prints_a_runnable_next_command(self, store_path):
        s = EnrollmentStore(store_path)
        s.add("a", emb(5, seed=10))
        s.add("b", emb(5, seed=11))
        s.save()
        result = runner.invoke(app, ["calibrate", "--store", str(store_path)])
        assert "tyf watch --match-threshold" in result.output


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info_runs_without_models_or_a_camera(self):
        """`tyf info` is the first thing to run when something is broken, so it
        must work in exactly the situation where nothing else does."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0, result.output
        for section in ("paths", "weights", "execution providers"):
            assert section in result.output

    def test_info_reports_whether_each_weight_is_cached(self):
        """The 'why does nothing work' answer is usually a missing download, so
        every model's state has to be stated, not implied."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "cached" in result.output or "not downloaded" in result.output
        assert "liveness" in result.output


# ---------------------------------------------------------------------------
# reporting: the numbers that end up in the README
# ---------------------------------------------------------------------------


class TestReportRendering:
    """`eval/report.py` formats every number this project publishes."""

    @staticmethod
    def _result():
        from trainyourface.core.contracts import AttackType
        from trainyourface.eval.report import evaluate_test

        scores = np.array([0.02, 0.05, 0.10, 0.20] + [0.95, 0.90, 0.30, 0.85])
        types = [AttackType.BONA_FIDE] * 4 + [AttackType.PRINT] * 2 + [AttackType.REPLAY] * 2
        return evaluate_test(scores, types, val_threshold=0.5)

    def test_report_leads_with_worst_case_apcer_and_states_n(self):
        from trainyourface.eval.report import render_report

        text = render_report(self._result())
        assert "APCER (worst)" in text
        assert "bona-fide" in text and "attack" in text
        assert "threshold" in text

    def test_worst_apcer_is_the_max_not_the_mean(self):
        """An attacker picks the attack, so the max is the honest aggregate."""
        r = self._result()
        by_type = r["deployed"]["apcer_by_type"]
        assert r["worst_apcer"] == max(by_type.values())
        assert r["worst_apcer"] >= r["mean_apcer"]

    def test_small_sample_warning_appears(self):
        from trainyourface.eval.report import render_report

        assert "n=8 is small" in render_report(self._result())

    def test_weakest_marker_is_omitted_when_all_types_tie(self):
        """Marking every row 'weakest' is noise; there is no weak spot to name."""
        from trainyourface.core.contracts import AttackType
        from trainyourface.eval.report import evaluate_test, render_report

        scores = np.array([0.1, 0.2, 0.9, 0.9])
        types = [AttackType.BONA_FIDE] * 2 + [AttackType.PRINT, AttackType.REPLAY]
        assert "weakest" not in render_report(evaluate_test(scores, types, 0.5))

    def test_weakest_marker_appears_when_one_type_is_worse(self):
        text_result = self._result()
        from trainyourface.eval.report import render_report

        assert "weakest" in render_report(text_result)

    def test_threshold_source_is_recorded(self):
        """The number in a report and the number the demo runs at must be tied
        together, so the report says where its threshold came from."""
        assert "validation" in self._result()["threshold_source"]

    def test_markdown_table_is_valid_and_complete(self):
        from trainyourface.eval.report import render_markdown_table

        table = render_markdown_table(self._result())
        rows = table.strip().split("\n")
        assert rows[0].startswith("|") and "---" in rows[1]
        # Every row is a well-formed 2-column markdown row.
        assert all(r.count("|") == 3 for r in rows)
        for metric in ("EER", "AUC", "BPCER", "APCER (worst case)", "ACER"):
            assert metric in table

    def test_saved_report_is_json_round_trippable(self, tmp_path):
        from trainyourface.eval.report import save_report

        path = tmp_path / "nested" / "report.json"
        save_report(self._result(), path)
        loaded = json.loads(path.read_text())
        assert loaded["worst_apcer"] == self._result()["worst_apcer"]

    def test_evaluate_test_does_not_recompute_the_threshold(self):
        """The threshold is an INPUT here. If test ever re-derived its own optimal
        point, the reported number would be 'the best we found by looking at the
        test labels', which is not a result."""
        from trainyourface.core.contracts import AttackType
        from trainyourface.eval.report import evaluate_test

        scores = np.array([0.02, 0.05, 0.95, 0.90])
        types = [AttackType.BONA_FIDE] * 2 + [AttackType.PRINT] * 2
        for thr in (0.3, 0.5, 0.7):
            assert evaluate_test(scores, types, val_threshold=thr)["threshold"] == thr


# ---------------------------------------------------------------------------
# pipeline construction: the "no liveness model" path
# ---------------------------------------------------------------------------


class TestLoaderWithoutLiveness:
    """A missing liveness model must degrade loudly, never quietly.

    Recognition-only is a legitimate mode — it's how a new user reaches a working
    demo before collecting PAD data — but in that mode a printed photo passes.
    The warning is the only thing standing between that and a user believing the
    tool checks liveness when it doesn't, so it's asserted rather than assumed.

    The detector and embedder are stubbed because they'd otherwise download
    weights; the liveness path under test is untouched.
    """

    @pytest.fixture
    def stub_models(self, monkeypatch):
        class Stub:
            spec = type("S", (), {"name": "stub"})()
            active_provider = "CPUExecutionProvider"

            def detect(self, image):
                return [], []

            def embed(self, crops):
                return np.zeros((len(crops), EMBEDDING_DIM), np.float32)

        monkeypatch.setattr("trainyourface.core.detect.FaceDetector", lambda *a, **k: Stub())
        monkeypatch.setattr("trainyourface.core.embed.FaceEmbedder", lambda *a, **k: Stub())

    def test_a_missing_model_warns_and_still_returns_a_pipeline(
        self, stub_models, tmp_path, capsys
    ):
        from trainyourface.cli.loader import load_pipeline

        pipeline = load_pipeline(
            liveness_path=tmp_path / "absent.onnx",
            store_path=tmp_path / "s.npz",
        )
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "cannot tell a real face" in err, "the warning must state the security impact"
        assert pipeline.liveness is None

    def test_no_face_is_trusted_without_a_liveness_model(self, stub_models, tmp_path):
        """Fail closed, end to end: the degraded pipeline cannot mark anything
        trustworthy, so the warning and the behaviour agree."""
        from trainyourface.cli.loader import load_pipeline

        pipeline = load_pipeline(
            liveness_path=tmp_path / "absent.onnx", store_path=tmp_path / "s.npz"
        )
        result = pipeline.process(np.zeros((240, 320, 3), np.uint8))
        assert all(not f.is_trustworthy for f in result.faces)

    def test_no_liveness_is_silent_because_it_was_asked_for(self, stub_models, tmp_path, capsys):
        """`--no-liveness` is an explicit choice; warning about a model the user
        told us not to load is noise. The `watch` command prints its own banner."""
        from trainyourface.cli.loader import load_pipeline

        load_pipeline(use_liveness=False, store_path=tmp_path / "s.npz")
        assert "WARNING" not in capsys.readouterr().err

    def test_quiet_suppresses_progress_but_not_the_warning(self, stub_models, tmp_path, capsys):
        """A security-relevant degradation is not progress chatter."""
        from trainyourface.cli.loader import load_pipeline

        load_pipeline(
            liveness_path=tmp_path / "absent.onnx",
            store_path=tmp_path / "s.npz",
            quiet=True,
        )
        err = capsys.readouterr().err
        assert "loading detector" not in err
        assert "WARNING" in err

    def test_recognition_can_be_skipped_entirely(self, stub_models, tmp_path):
        from trainyourface.cli.loader import load_pipeline

        pipeline = load_pipeline(recognize=False, use_liveness=False)
        assert pipeline.embedder is None
        assert pipeline.store is None
