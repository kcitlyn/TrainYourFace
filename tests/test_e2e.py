"""End-to-end: synthetic dataset -> manifest -> train -> eval -> export -> predict.

Everything else in this suite tests one stage in isolation. This file runs the
whole pipeline the way the README tells a user to, because the failures that
actually matter live in the SEAMS between stages, and no unit test can see them:

  - train writes a checkpoint whose config `eval` must reconstruct exactly
  - train picks a threshold on val that `eval` must reuse unchanged
  - export must produce an ONNX graph agreeing with the PyTorch model
  - the ONNX model must run through the same preprocessing as training

A train/serve preprocessing mismatch is the canonical example: both sides pass
their own tests and the system is still broken. The previous version of this
project shipped exactly that bug in its recognition path for months.

WHY THE DATA IS SYNTHETIC
-------------------------
The "bona-fide" images are smooth gradients and the "attacks" carry a high
frequency grid — a crude stand-in for the moiré a real screen replay produces.
This is NOT a claim about real-world accuracy. It is a learnable signal, which is
all that's needed to check that gradients flow, the threshold is carried through,
and the exported model agrees with the trained one. Real numbers require the real
captured dataset; see the README.

These tests need torch, so they carry the `train` marker and are excluded from the
default run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from trainyourface.cli.main import app
from trainyourface.core.contracts import AttackType
from trainyourface.liveness.dataset import (
    INPUT_SIZE,
    DatasetManifest,
    Sample,
    split_fingerprint,
    subject_disjoint_split,
    verify_split_matches,
)

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.train

runner = CliRunner()

N_SUBJECTS = 6
PER_CLASS = 8


def _bona_fide(rng: np.random.Generator) -> np.ndarray:
    """A smooth gradient: low-frequency content only, like a real face."""
    ramp = np.linspace(60, 200, INPUT_SIZE, dtype=np.float32)
    img = ramp[None, :] * 0.5 + ramp[:, None] * 0.5
    img = img + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


def _attack(rng: np.random.Generator) -> np.ndarray:
    """The same gradient plus a high-frequency grid — a stand-in for moiré."""
    base = _bona_fide(rng).astype(np.float32)
    yy, xx = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
    grid = 26.0 * np.sin(xx * 1.9) * np.sin(yy * 1.9)
    return np.clip(base + grid[:, :, None], 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A synthetic PAD dataset on disk, exactly as `tyf capture` would leave it."""
    import cv2

    root = tmp_path_factory.mktemp("pad")
    rng = np.random.default_rng(0)
    samples = []

    for i in range(N_SUBJECTS):
        subject = f"subj{i:02d}"
        for attack_type in (AttackType.BONA_FIDE, AttackType.PRINT, AttackType.REPLAY):
            for j in range(PER_CLASS):
                rel = f"images/{attack_type.value}/{subject}_{j:03d}.png"
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                img = _bona_fide(rng) if attack_type is AttackType.BONA_FIDE else _attack(rng)
                assert cv2.imwrite(str(path), img)
                samples.append(
                    Sample(
                        path=rel,
                        attack_type=attack_type,
                        subject=subject,
                        session="s1",
                        instrument=None if attack_type is AttackType.BONA_FIDE else "synthetic",
                    )
                )

    DatasetManifest(samples=samples, root=str(root)).save(root / "manifest.json")
    return root


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    """Run `tyf train` for real, once, and hand the run directory to every test."""
    out = tmp_path_factory.mktemp("runs") / "liveness"
    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(dataset),
            "--out",
            str(out),
            "--epochs",
            "6",
            "--batch-size",
            "16",
            "--width",
            "16",
            "--device",
            "cpu",
            "--seed",
            "0",
        ],
    )
    if result.exit_code != 0:  # pragma: no cover - surfaces the real traceback
        raise AssertionError(
            f"train failed ({result.exit_code}):\n{result.output}\n{result.exception}"
        )
    return out, result.output


@pytest.fixture(scope="module")
def exported(trained):
    """Run `tyf export` for real, once. Returns the ONNX path and the output."""
    out, _ = trained
    result = runner.invoke(
        app, ["export", "--checkpoint", str(out / "best.pt"), "--format", "onnx"]
    )
    if result.exit_code != 0:  # pragma: no cover
        raise AssertionError(f"export failed:\n{result.output}\n{result.exception}")
    # The exported name is the one `default_model_path()` looks for, so asserting
    # it here also pins the contract between export and the deployed lookup: a
    # rename would leave `tyf watch` silently unable to find the model it just
    # exported.
    path = out / "liveness.onnx"
    assert path.exists(), result.output
    return path, result.output


# ---------------------------------------------------------------------------
# stage 1: the dataset is well-formed and splittable
# ---------------------------------------------------------------------------


class TestDatasetStage:
    def test_manifest_command_accepts_the_dataset(self, dataset):
        result = runner.invoke(app, ["manifest", "--data", str(dataset)])
        assert result.exit_code == 0, result.output
        assert f"{N_SUBJECTS * PER_CLASS * 3} samples, {N_SUBJECTS} subjects" in result.output
        assert "images present" in result.output

    def test_the_split_is_subject_disjoint(self, dataset):
        m = DatasetManifest.load(dataset / "manifest.json")
        train, val, test = subject_disjoint_split(m.samples, seed=0)
        groups = [{s.subject for s in part} for part in (train, val, test)]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]
        assert set().union(*groups) == set(m.subjects()), "a subject was dropped entirely"

    def test_every_split_has_both_classes(self, dataset):
        """PAD metrics are undefined on a single-class split."""
        m = DatasetManifest.load(dataset / "manifest.json")
        for part in subject_disjoint_split(m.samples, seed=0):
            assert {s.label for s in part} == {0, 1}


# ---------------------------------------------------------------------------
# stage 2: training
# ---------------------------------------------------------------------------


class TestTrainStage:
    def test_train_writes_a_checkpoint_and_a_summary(self, trained):
        out, _ = trained
        assert (out / "best.pt").exists()
        assert (out / "train_summary.json").exists()
        assert (out / "test_report.json").exists()

    def test_the_model_actually_learned_the_signal(self, trained):
        """Not a benchmark claim — a check that gradients flowed at all.

        A model that failed to learn sits at EER ~= 0.5. The synthetic moiré cue
        is trivially separable, so anything above 0.35 means the training loop is
        broken, not that the task is hard.
        """
        out, _ = trained
        summary = json.loads((out / "train_summary.json").read_text())
        assert summary["best_val_eer"] < 0.35, (
            f"training did not converge: {summary['best_val_eer']}"
        )

    def test_the_threshold_comes_from_validation_and_is_recorded(self, trained):
        out, _ = trained
        summary = json.loads((out / "train_summary.json").read_text())
        assert 0.0 <= summary["val_threshold"] <= 1.0

    def test_train_reports_the_split_as_verified_disjoint(self, trained):
        _, output = trained
        assert "disjoint, verified" in output

    def test_the_test_report_uses_the_validation_threshold_unchanged(self, trained):
        """The core honesty property of the whole pipeline.

        If test ever re-derived its own threshold, the headline number would be
        "the best operating point we found by looking at the test labels", which
        is not a result. Asserting equality here is what makes that hard to
        regress.
        """
        out, _ = trained
        summary = json.loads((out / "train_summary.json").read_text())
        report = json.loads((out / "test_report.json").read_text())
        assert report["threshold"] == pytest.approx(summary["val_threshold"])
        assert "validation" in report["threshold_source"]

    def test_the_test_report_is_complete(self, trained):
        out, _ = trained
        report = json.loads((out / "test_report.json").read_text())
        for key in ("eer", "auc", "worst_apcer", "bpcer", "acer", "bpcer_at_apcer"):
            assert key in report, f"{key} missing from the report"
        assert report["worst_apcer"] >= report["mean_apcer"]
        assert report["n_bona_fide"] > 0 and report["n_attack"] > 0

    def test_per_attack_type_apcer_is_reported_for_both_types(self, trained):
        """A single averaged APCER would hide a model that catches prints and
        misses replays."""
        out, _ = trained
        report = json.loads((out / "test_report.json").read_text())
        by_type = report["deployed"]["apcer_by_type"]
        assert set(by_type) == {"print", "replay"}

    def test_untested_attack_types_are_absent_not_zero(self, trained):
        """Claiming 0% against an attack never tested is the exact kind of
        flattering number this project exists to avoid."""
        out, _ = trained
        report = json.loads((out / "test_report.json").read_text())
        assert "mask_3d" not in report["deployed"]["apcer_by_type"]

    def test_the_split_fingerprint_is_persisted(self, trained):
        out, _ = trained
        split = json.loads((out / "train_summary.json").read_text())["split"]
        assert split["by"] == "subject"
        assert split["test"] and not set(split["test"]) & set(split["train"])

    def test_the_persisted_fingerprint_matches_a_fresh_reconstruction(self, dataset, trained):
        """What `tyf eval` relies on, checked against the real training artifact.

        The fingerprint on disk has to agree with what re-deriving the split from
        the manifest produces, or the verification in `eval` would reject every
        legitimate run.
        """
        out, _ = trained
        stored = json.loads((out / "train_summary.json").read_text())["split"]

        m = DatasetManifest.load(dataset / "manifest.json")
        parts = subject_disjoint_split(m.samples, seed=0, by="subject")
        assert split_fingerprint(*parts, by="subject") == stored
        verify_split_matches(stored, parts[2], by="subject")  # must not raise


# ---------------------------------------------------------------------------
# stage 3: re-evaluating the checkpoint
# ---------------------------------------------------------------------------


class TestEvalStage:
    def test_eval_reproduces_the_numbers_train_reported(self, dataset, trained):
        """Same checkpoint, same data, same seed -> identical metrics.

        This is the reproducibility claim in the README, checked rather than
        asserted. Any nondeterminism in preprocessing or ordering breaks it.
        """
        out, _ = trained
        result = runner.invoke(
            app,
            ["eval", "--checkpoint", str(out / "best.pt"), "--data", str(dataset), "--seed", "0"],
        )
        assert result.exit_code == 0, result.output

        report = json.loads((out / "test_report.json").read_text())
        assert f"{report['eer'] * 100:6.2f}%" in result.output
        assert f"{report['auc']:6.4f}" in result.output

    def test_eval_confirms_the_split_matches_training(self, dataset, trained):
        out, _ = trained
        result = runner.invoke(
            app, ["eval", "--checkpoint", str(out / "best.pt"), "--data", str(dataset)]
        )
        assert "matches training" in result.output

    def test_a_mismatched_seed_is_refused_as_a_split_mismatch(self, dataset, trained):
        """The leak this catches is invisible otherwise.

        `--seed` defaults to 0 regardless of what training used, so a wrong seed
        re-partitions the data and puts subjects the model TRAINED ON into the
        "held-out" test set. The resulting split is still internally disjoint, so
        the integrity check passes and the report looks entirely normal — the
        numbers are just inflated. Only comparing against the recorded split
        catches it.
        """
        out, _ = trained
        result = runner.invoke(
            app,
            ["eval", "--checkpoint", str(out / "best.pt"), "--data", str(dataset), "--seed", "7"],
        )
        assert result.exit_code == 1
        assert "SPLIT MISMATCH" in result.output

    def test_a_mismatched_split_mode_is_refused(self, dataset, trained):
        out, _ = trained
        result = runner.invoke(
            app,
            [
                "eval",
                "--checkpoint",
                str(out / "best.pt"),
                "--data",
                str(dataset),
                "--split-by",
                "session",
            ],
        )
        assert result.exit_code == 1
        assert "SPLIT MISMATCH" in result.output

    def test_markdown_output_is_readme_ready(self, dataset, trained):
        out, _ = trained
        result = runner.invoke(
            app,
            [
                "eval",
                "--checkpoint",
                str(out / "best.pt"),
                "--data",
                str(dataset),
                "--markdown",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "| Metric | Value |" in result.output
        assert "APCER (worst case)" in result.output

    def test_an_explicit_threshold_overrides_the_stored_one(self, dataset, trained):
        out, _ = trained
        result = runner.invoke(
            app,
            [
                "eval",
                "--checkpoint",
                str(out / "best.pt"),
                "--data",
                str(dataset),
                "--threshold",
                "0.9",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "0.9000" in result.output


# ---------------------------------------------------------------------------
# stage 4: export, and the train/serve agreement
# ---------------------------------------------------------------------------


class TestExportStage:
    def test_export_verifies_against_pytorch_by_default(self, exported):
        """`--verify` is on by default because a silently wrong export is the
        worst outcome: the model ships, runs, and is subtly incorrect. The
        verification is part of the command, so its result belongs in the output.
        """
        _, output = exported
        assert "verified" in output
        assert "max |onnx - torch|" in output

    def test_export_reports_the_artifact_size(self, exported):
        """Size is the deployability number on a Pi, so it's printed, not implied."""
        _, output = exported
        assert "MB" in output

    def test_the_onnx_model_agrees_with_the_pytorch_model(self, exported, trained):
        """The train/serve seam, checked directly on real data.

        Both sides go through `to_model_input`, the one shared preprocessing
        function, which is the point: this is the bug class that took down the
        previous version of this project's recognition path.
        """
        import onnxruntime as ort

        from trainyourface.liveness.dataset import to_model_input
        from trainyourface.liveness.model import ModelConfig, build_model

        out, _ = trained
        ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        rng = np.random.default_rng(99)
        batch = np.stack([to_model_input(_attack(rng)) for _ in range(4)])

        with torch.no_grad():
            torch_out = torch.softmax(model(torch.from_numpy(batch)), dim=1).numpy()

        sess = ort.InferenceSession(str(exported[0]), providers=["CPUExecutionProvider"])
        onnx_logits = sess.run(None, {sess.get_inputs()[0].name: batch})[0]
        exp = np.exp(onnx_logits - onnx_logits.max(axis=1, keepdims=True))
        onnx_out = exp / exp.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(torch_out, onnx_out, atol=1e-4)

    def test_the_exported_model_accepts_a_dynamic_batch(self, exported):
        """The live path scores a variable number of faces per frame, so a graph
        frozen at the export batch size would fail on the second face."""
        import onnxruntime as ort

        from trainyourface.liveness.dataset import to_model_input

        sess = ort.InferenceSession(str(exported[0]), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        rng = np.random.default_rng(7)
        for n in (1, 3, 8):
            batch = np.stack([to_model_input(_bona_fide(rng)) for _ in range(n)])
            assert sess.run(None, {name: batch})[0].shape[0] == n

    def test_bench_runs_on_the_exported_model(self, exported):
        result = runner.invoke(
            app, ["bench", "--model", str(exported[0]), "--runs", "10", "--warmup", "2"]
        )
        assert result.exit_code == 0, result.output
        assert "p95" in result.output.lower()


# ---------------------------------------------------------------------------
# stage 5: inference — the deployed path
# ---------------------------------------------------------------------------


class TestPredictStage:
    def test_the_deployed_model_loads_the_validation_threshold(self, exported, trained):
        """The number reported and the number enforced at runtime must be the
        same one, or the report describes a system nobody is running."""
        from trainyourface.liveness.predict import LivenessModel

        out, _ = trained
        summary = json.loads((out / "train_summary.json").read_text())
        model = LivenessModel(exported[0])
        assert model.threshold == pytest.approx(summary["val_threshold"])

    def test_it_ranks_synthetic_attacks_above_bona_fide(self, exported):
        """End of the chain: the artifact on disk still discriminates.

        Asserted as RANKING (every attack scores above every genuine sample),
        not as an absolute probability gap. Those are different properties, and
        conflating them is easy to do by accident: this model separates the two
        classes perfectly while both means sit near 0.12, because six epochs of
        label-smoothed training on a tiny synthetic set produces an
        under-confident but correctly-ordered model. A margin assertion would
        fail on a model that is in fact working, which makes it a test of
        calibration wearing a discrimination test's clothes.

        Ranking is also the property the metrics depend on — EER and AUC are both
        rank statistics, and the threshold is derived from the score
        distribution rather than assumed.
        """
        from trainyourface.liveness.predict import LivenessModel

        model = LivenessModel(exported[0])
        rng = np.random.default_rng(123)
        live = model.spoof_probability([_bona_fide(rng) for _ in range(8)])
        spoof = model.spoof_probability([_attack(rng) for _ in range(8)])

        assert np.all((live >= 0.0) & (live <= 1.0)), "scores must be probabilities"
        assert np.all((spoof >= 0.0) & (spoof <= 1.0))
        assert spoof.min() > live.max(), (
            f"the exported model does not separate the classes: "
            f"live=[{live.min():.3f}, {live.max():.3f}] "
            f"spoof=[{spoof.min():.3f}, {spoof.max():.3f}]"
        )

    def test_the_deployed_threshold_is_a_real_decision_boundary(self, exported):
        """Ranking alone is not enough: a threshold outside the score range would
        call everything live (or everything a spoof) while the ranking test still
        passed. This checks the val-derived threshold lands inside the range where
        the two classes actually meet.

        Deliberately NOT asserting perfect classification on these samples. The
        threshold was chosen on held-out validation subjects and applied
        unchanged, so a few errors on fresh data are expected — that is what
        generalization looks like, and an assertion that forbade it would be a
        benchmark claim this synthetic data cannot support. Asserting the
        boundary is *positioned* correctly is the integrity property; how many
        samples fall on the wrong side of it is what the reported BPCER/APCER are
        for.
        """
        from trainyourface.liveness.predict import LivenessModel

        model = LivenessModel(exported[0])
        rng = np.random.default_rng(321)
        live = model.spoof_probability([_bona_fide(rng) for _ in range(8)])
        spoof = model.spoof_probability([_attack(rng) for _ in range(8)])

        assert live.min() < model.threshold < spoof.max(), (
            f"threshold {model.threshold:.4f} sits outside the range where the "
            f"classes meet: live=[{live.min():.4f}, {live.max():.4f}] "
            f"spoof=[{spoof.min():.4f}, {spoof.max():.4f}]"
        )
        # And it is on the right side of each class's centre of mass.
        assert live.mean() < model.threshold < spoof.mean()

    def test_error_rates_at_the_deployed_threshold_beat_chance(self, exported):
        """The weaker but honest accuracy claim: both error rates are well below
        50%. A model that learned nothing would sit at chance on one of them."""
        from trainyourface.liveness.predict import LivenessModel

        model = LivenessModel(exported[0])
        rng = np.random.default_rng(654)
        live = model.spoof_probability([_bona_fide(rng) for _ in range(16)])
        spoof = model.spoof_probability([_attack(rng) for _ in range(16)])

        bpcer = float((live >= model.threshold).mean())
        apcer = float((spoof < model.threshold).mean())
        assert bpcer < 0.25, f"BPCER {bpcer:.2f} — genuine samples wrongly rejected"
        assert apcer < 0.25, f"APCER {apcer:.2f} — attacks wrongly accepted"

    def test_an_empty_batch_returns_no_scores(self, exported):
        """A frame with no faces is normal, not an error."""
        from trainyourface.liveness.predict import LivenessModel

        assert LivenessModel(exported[0]).spoof_probability([]).shape == (0,)

    def test_verdicts_carry_the_threshold_they_were_decided_with(self, exported):
        """A score without its threshold cannot be audited after the fact."""
        from trainyourface.core.contracts import Box
        from trainyourface.liveness.predict import LivenessModel

        model = LivenessModel(exported[0])
        rng = np.random.default_rng(5)
        frame = _attack(rng)
        h, w = frame.shape[:2]
        result = model.check(frame, [Box(x1=0, y1=0, x2=w, y2=h, score=0.99)])[0]
        assert result.threshold == model.threshold
        assert result.is_live == (result.spoof_probability < model.threshold)
        assert result.label == ("REAL" if result.is_live else "SPOOF")

    def test_the_model_reports_where_its_threshold_came_from(self, exported):
        """Provenance, not just the value: 'validation split' and 'I picked 0.5'
        are very different claims to publish."""
        from trainyourface.liveness.predict import LivenessModel

        assert "validation" in LivenessModel(exported[0]).threshold_source
        override = LivenessModel(exported[0], threshold=0.8)
        assert override.threshold == 0.8
        assert "override" in override.threshold_source


# ---------------------------------------------------------------------------
# stage 6: the pipeline as a whole, with recognition gated on liveness
# ---------------------------------------------------------------------------


class TestGatedPipeline:
    """Liveness gates recognition, using the real trained model.

    This is the project's actual security claim, exercised against a real
    artifact rather than a stub: a face that fails liveness must not be named.
    """

    def _pipeline(self, onnx_path, store):
        from trainyourface.core.contracts import EMBEDDING_DIM, Box
        from trainyourface.core.pipeline import FacePipeline
        from trainyourface.liveness.predict import LivenessModel

        class Detector:
            spec = type("S", (), {"name": "stub"})()
            active_provider = "CPUExecutionProvider"

            def detect(self, image):
                h, w = image.shape[:2]
                return [Box(x1=0, y1=0, x2=w, y2=h, score=0.99)], [np.zeros((5, 2), np.float32)]

        class Embedder:
            spec = type("S", (), {"name": "stub"})()
            active_provider = "CPUExecutionProvider"

            def embed(self, crops):
                v = np.zeros((len(crops), EMBEDDING_DIM), np.float32)
                v[:, 0] = 1.0
                return v

        return FacePipeline(
            detector=Detector(),
            embedder=Embedder(),
            liveness=LivenessModel(onnx_path),
            store=store,
        )

    @pytest.fixture
    def store(self, tmp_path):
        from trainyourface.core.contracts import EMBEDDING_DIM
        from trainyourface.core.store import EnrollmentStore

        s = EnrollmentStore(tmp_path / "enroll.npz")
        v = np.zeros((3, EMBEDDING_DIM), np.float32)
        v[:, 0] = 1.0
        s.add("kaitlyn", v)
        return s

    def test_a_live_face_is_identified(self, exported, store):
        rng = np.random.default_rng(1)
        result = self._pipeline(exported[0], store).process(_bona_fide(rng))
        assert len(result.faces) == 1
        face = result.faces[0]
        assert face.liveness.is_live, f"bona-fide scored {face.liveness.spoof_probability:.3f}"
        assert face.identity.name == "kaitlyn"
        assert face.is_trustworthy

    def test_a_spoofed_face_is_never_named(self, exported, store):
        """Recognition is gated on liveness, so a spoof carries no identity at
        all. Naming it would confirm to an attacker that they targeted the right
        person."""
        rng = np.random.default_rng(2)
        result = self._pipeline(exported[0], store).process(_attack(rng))
        assert len(result.faces) == 1
        face = result.faces[0]
        assert not face.liveness.is_live, f"attack scored {face.liveness.spoof_probability:.3f}"
        assert face.identity is None
        assert not face.is_trustworthy

    def test_unknown_liveness_is_untrusted(self, exported, store):
        """Fail closed. A missing verdict is not a passing one."""
        rng = np.random.default_rng(3)
        pipeline = self._pipeline(exported[0], store)
        pipeline.liveness = None
        face = pipeline.process(_bona_fide(rng)).faces[0]
        assert face.liveness is None
        assert not face.is_trustworthy, "no liveness check must never read as trustworthy"
