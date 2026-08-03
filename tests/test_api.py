"""Tests for the public Python API and the demographic fairness audit.

The API tests care about one thing above correctness of the happy path: the
DEFAULTS must be the safe ones. This is a security library, and the difference
between `FaceID()` refusing to run without a liveness model and quietly falling
back to recognition-only is the difference between a tool that protects you and a
tool that looks like it does.

The fairness tests exist because the audit makes a stronger kind of claim than the
rest of the project — "this model treats groups differently" — and a bias metric
that is subtly wrong is worse than no metric, because it gets quoted.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.contracts import AttackType
from trainyourface.core.embed import EMBEDDING_DIM
from trainyourface.eval.report import (
    DISPARATE_IMPACT_RATIO,
    MIN_FAIRNESS_GROUP,
    evaluate_test,
    fairness_audit,
    render_report,
    stratify,
)
from trainyourface.liveness.celeba_spoof import (
    CELEBA_ATTRIBUTES,
    FAIRNESS_ATTRIBUTES,
    parse_label,
)


def vec(spoof_type: int = 0, illum: int = 0, env: int = 0, **attrs) -> list:
    """A 44-element label vector with named CelebA attributes set to +1.

    Attributes default to -1 (absent), matching CelebA's own +1/-1 encoding rather
    than 1/0 — the distinction the parser has to get right.
    """
    v = [-1] * 40 + [spoof_type, illum, env, int(spoof_type != 0)]
    for name, present in attrs.items():
        v[CELEBA_ATTRIBUTES.index(name)] = 1 if present else -1
    return v


class TestAttributeParsing:
    def test_attributes_are_off_by_default(self):
        """A manifest nobody will audit shouldn't carry 10 extra keys per sample."""
        _, cond = parse_label(vec())
        assert not any(k.startswith("attr:") for k in cond)

    def test_minus_one_means_absent_not_present(self):
        """CelebA encodes absence as -1, which is TRUTHY in Python.

        A naive `bool(raw)` would mark every attribute present on every face,
        producing a fairness report with one bucket per axis, no disparity anywhere,
        and a clean bill of health manufactured entirely by a sign error.
        """
        _, cond = parse_label(vec(Eyeglasses=False), attributes=True)
        assert cond["attr:Eyeglasses"] == "no"

    def test_plus_one_means_present(self):
        _, cond = parse_label(vec(Eyeglasses=True), attributes=True)
        assert cond["attr:Eyeglasses"] == "yes"

    def test_only_the_fairness_subset_is_recorded(self):
        _, cond = parse_label(vec(), attributes=True)
        recorded = {k[5:] for k in cond if k.startswith("attr:")}
        assert recorded == set(FAIRNESS_ATTRIBUTES)

    def test_attractive_is_excluded(self):
        """A subjective crowd annotation is not a category to report rates by."""
        assert "Attractive" not in FAIRNESS_ATTRIBUTES
        assert "Attractive" in CELEBA_ATTRIBUTES, "still parsed, just not audited"

    def test_blurry_is_excluded(self):
        """Image quality is a robustness axis, not a property of the person."""
        assert "Blurry" not in FAIRNESS_ATTRIBUTES

    def test_the_attribute_list_is_the_documented_length(self):
        assert len(CELEBA_ATTRIBUTES) == 40
        assert len(set(CELEBA_ATTRIBUTES)) == 40, "a duplicate would shift every index"

    def test_every_audited_attribute_exists_in_the_canonical_list(self):
        """A typo here would raise ValueError deep in a 625K-image conversion."""
        for name in FAIRNESS_ATTRIBUTES:
            assert name in CELEBA_ATTRIBUTES

    def test_attributes_are_recorded_on_live_samples(self):
        """The whole audit depends on this.

        Capture conditions are empty for live images, which is why APCER is the
        only thing `stratify` can report. BPCER disparity needs labels on BONA
        FIDE samples — if attributes were spoof-only, this feature couldn't exist.
        """
        attack, cond = parse_label(vec(spoof_type=0, Male=True), attributes=True)
        assert attack is AttackType.BONA_FIDE
        assert cond["attr:Male"] == "yes"


def _bona_fide_groups(spec: dict[str, tuple[int, int]], threshold: float = 0.5):
    """Build bona-fide samples: {group_value: (n_total, n_rejected)}."""
    scores, types, conds = [], [], []
    for value, (n, rejected) in spec.items():
        for i in range(n):
            # score >= threshold == scored as attack == wrongly rejected
            scores.append(0.9 if i < rejected else 0.1)
            types.append(AttackType.BONA_FIDE)
            conds.append({"attr:Eyeglasses": value})
    return np.array(scores), types, conds


class TestFairnessAudit:
    def test_it_measures_bpcer_not_apcer(self):
        """Fairness is about who gets wrongly REJECTED.

        APCER harm lands on the account owner; BPCER harm lands on the user, every
        time they try to unlock. Auditing APCER by demographic would answer a
        question nobody asked.
        """
        n = MIN_FAIRNESS_GROUP
        scores, types, conds = _bona_fide_groups({"yes": (n, n // 2), "no": (n, 0)})
        out = fairness_audit(scores, types, conds, threshold=0.5)
        assert out["Eyeglasses"]["groups"]["yes"]["bpcer"] == pytest.approx(0.5)
        assert out["Eyeglasses"]["groups"]["no"]["bpcer"] == pytest.approx(0.0)

    def test_attack_samples_are_excluded(self):
        """Mixing them would blend two different rates into one meaningless number."""
        out = fairness_audit(
            np.array([0.9, 0.9]),
            [AttackType.BONA_FIDE, AttackType.REPLAY],
            [{"attr:Male": "yes"}, {"attr:Male": "yes"}],
            threshold=0.5,
        )
        assert out["Male"]["groups"]["yes"]["n"] == 1

    def test_capture_conditions_are_not_audited(self):
        """A disparity by screen bezel is not a fairness finding."""
        out = fairness_audit(
            np.array([0.9] * 3),
            [AttackType.BONA_FIDE] * 3,
            [{"illumination": "back"}] * 3,
            threshold=0.5,
        )
        assert out == {}

    def test_a_disparity_over_the_four_fifths_ratio_is_flagged(self):
        n = MIN_FAIRNESS_GROUP
        scores, types, conds = _bona_fide_groups(
            {"yes": (n, n // 2), "no": (n, n // 10)}  # 50% vs 10% = 5x
        )
        out = fairness_audit(scores, types, conds, threshold=0.5)
        assert out["Eyeglasses"]["flagged"] is True
        assert out["Eyeglasses"]["ratio"] == pytest.approx(5.0)

    def test_a_small_disparity_is_not_flagged(self):
        """The threshold must not fire on noise, or every report reads as biased."""
        n = MIN_FAIRNESS_GROUP * 4
        # 10% vs 9% -> ratio 1.11, under the 1.25 screening threshold
        scores, types, conds = _bona_fide_groups({"yes": (n, n // 10), "no": (n, int(n * 0.09))})
        out = fairness_audit(scores, types, conds, threshold=0.5)
        assert out["Eyeglasses"]["ratio"] < DISPARATE_IMPACT_RATIO
        assert out["Eyeglasses"]["flagged"] is False

    def test_a_zero_baseline_is_flagged_without_an_infinite_ratio(self):
        """0% in one group and 8% in another is a real disparity.

        The ratio is undefined (division by zero), which must not become `inf` in
        the JSON or crash the renderer. The absolute gap carries the finding.
        """
        n = MIN_FAIRNESS_GROUP
        scores, types, conds = _bona_fide_groups({"yes": (n, 4), "no": (n, 0)})
        out = fairness_audit(scores, types, conds, threshold=0.5)
        entry = out["Eyeglasses"]
        assert entry["ratio"] is None
        assert entry["flagged"] is True
        assert entry["gap"] == pytest.approx(4 / n)

    def test_small_groups_get_no_rate(self):
        """ "This model is biased" from n=25 is irresponsible."""
        scores, types, conds = _bona_fide_groups({"yes": (10, 5), "no": (10, 0)})
        out = fairness_audit(scores, types, conds, threshold=0.5)
        groups = out["Eyeglasses"]["groups"]
        assert groups["yes"]["bpcer"] is None
        assert groups["yes"]["rejected"] == 5, "the raw count is still reported"
        assert "ratio" not in out["Eyeglasses"], "no ratio from unrateable groups"

    def test_the_fairness_floor_is_stricter_than_the_robustness_floor(self):
        """A bias claim carries more weight than a robustness observation."""
        from trainyourface.eval.report import MIN_BUCKET

        assert MIN_FAIRNESS_GROUP > MIN_BUCKET

    def test_a_single_group_produces_no_ratio(self):
        """Nothing to compare against; a ratio would be invented from one number."""
        n = MIN_FAIRNESS_GROUP
        scores, types, conds = _bona_fide_groups({"yes": (n, n // 2)})
        out = fairness_audit(scores, types, conds, threshold=0.5)
        assert "ratio" not in out["Eyeglasses"]

    def test_misaligned_types_and_conditions_raise(self):
        with pytest.raises(ValueError, match="misaligned"):
            fairness_audit(
                np.array([0.1]),
                [AttackType.BONA_FIDE],
                [{}, {}],  # two conditions, one type
                threshold=0.5,
            )

    def test_wrong_score_count_raises(self):
        with pytest.raises(ValueError, match="expected 1 scores"):
            fairness_audit(np.array([0.1, 0.2]), [AttackType.BONA_FIDE], [{}], threshold=0.5)

    def test_a_nan_score_raises_rather_than_undercounting(self):
        """A NaN reaches `score >= threshold` as a silent False.

        That would count a non-finite score as "not rejected" and understate
        BPCER — a flattering wrong number, exactly what the metrics layer rejects
        with a raise. This guard makes the public function behave the same way its
        callers do, instead of only when reached through evaluate_test.
        """
        n = MIN_FAIRNESS_GROUP
        scores = np.array([np.nan] + [0.1] * (n - 1))
        with pytest.raises(ValueError, match="NaN or inf"):
            fairness_audit(scores, [AttackType.BONA_FIDE] * n, [{"attr:Male": "yes"}] * n, 0.5)

    def test_an_out_of_range_score_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            fairness_audit(
                np.array([1.5] * 3), [AttackType.BONA_FIDE] * 3, [{"attr:Male": "yes"}] * 3, 0.5
            )


class TestFairnessRendering:
    @staticmethod
    def _result():
        n = MIN_FAIRNESS_GROUP
        scores, types, conds = _bona_fide_groups({"yes": (n, n // 2), "no": (n, n // 10)})
        # Attacks so EER/BPCER are computable.
        scores = np.concatenate([scores, np.full(n, 0.95)])
        types = types + [AttackType.REPLAY] * n
        conds = conds + [{"attr:Pale_Skin": "yes"}] * n
        return evaluate_test(scores, types, 0.5, conditions=conds)

    def test_the_section_appears_with_the_disparity_first(self):
        text = render_report(self._result())
        assert "BPCER by demographic group" in text
        assert "disparity" in text
        # The worse group must render above the better one. Matched on the data
        # ROWS rather than on the bare strings "yes"/"no": the surrounding prose
        # contains "not", so a substring search finds a false position and the
        # assertion passes or fails for the wrong reason.
        rows = [
            ln.strip().split()[0]
            for ln in text.splitlines()
            if ln.startswith("      ") and "%" in ln and "@" not in ln
        ]
        assert rows[:2] == ["yes", "no"], f"expected worst-first, got {rows[:2]}"

    def test_the_four_fifths_rule_is_explained_not_just_applied(self):
        """A flag nobody can interpret is noise; the standard is named."""
        flat = " ".join(render_report(self._result()).split())
        assert "four-fifths" in flat
        assert "not a finding" in flat, "a flag is a prompt to investigate"

    def test_the_pale_skin_caveat_prints_when_that_attribute_is_present(self):
        """The most important line in the whole feature.

        Pale_Skin is a binary crowd annotation, not a validated skin-tone measure.
        Reporting a disparity along it as "skin-tone bias" would be exactly the
        overclaim this project refuses, so the caveat travels with the number.
        """
        n = MIN_FAIRNESS_GROUP
        scores, types, conds = [], [], []
        for value, rejected in (("yes", n // 2), ("no", 0)):
            for i in range(n):
                scores.append(0.9 if i < rejected else 0.1)
                types.append(AttackType.BONA_FIDE)
                conds.append({"attr:Pale_Skin": value})
        scores += [0.95] * n
        types += [AttackType.REPLAY] * n
        conds += [{}] * n
        text = render_report(evaluate_test(np.array(scores), types, 0.5, conditions=conds))
        flat = " ".join(text.split())
        assert "not a validated skin-tone measurement" in flat
        assert "Fitzpatrick" in text and "Monk" in text

    def test_no_attributes_means_no_section(self):
        """A webcam dataset has no demographic labels and must not grow a header."""
        result = evaluate_test(
            np.array([0.9] * 5 + [0.1] * 5),
            [AttackType.REPLAY] * 5 + [AttackType.BONA_FIDE] * 5,
            0.5,
            conditions=[{"illumination": "back"}] * 5 + [{}] * 5,
        )
        assert "fairness" not in result
        assert "demographic" not in render_report(result)

    def test_attributes_do_not_leak_into_the_condition_table(self):
        """`attr:` keys belong to the fairness audit only.

        Without the exclusion, `stratify` would report "APCER among people wearing
        glasses" in the capture-condition table — a real number that reads as a
        demographic finding while actually being about attack samples.
        """
        out = stratify(
            np.array([0.1] * 5),
            [AttackType.REPLAY] * 5,
            [{"attr:Male": "yes", "illumination": "back"}] * 5,
            threshold=0.5,
        )
        assert "illumination" in out
        assert not any(k.startswith("attr:") or k == "Male" for k in out)


def _observation(name=None, is_live=None, embedding=True, box=(0, 0, 40, 40)):
    """A FaceObservation as the real pipeline would emit one."""
    from trainyourface.core.contracts import Box, FaceObservation, Identity, LivenessResult

    liveness = None
    if is_live is not None:
        # spoof_probability below threshold == live, matching LivenessResult.is_live
        liveness = LivenessResult(
            spoof_probability=0.1 if is_live else 0.9,
            threshold=0.5,
        )
    return FaceObservation(
        box=Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3], score=0.99),
        embedding=(np.ones(EMBEDDING_DIM, np.float32) if embedding else None),
        identity=(Identity(name=name, similarity=0.9) if name else None),
        liveness=liveness,
    )


class _StubPipeline:
    """Stands in for FacePipeline, counting how many times it runs.

    The count is the point for one of these tests: `enroll` used to call the
    pipeline twice, which made the liveness verdict it enforced belong to a
    different inference than the embedding it stored.
    """

    def __init__(self, observations):
        self._observations = observations
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        from trainyourface.core.pipeline import FrameResult

        return FrameResult(faces=list(self._observations))


@pytest.fixture
def stub_faceid(tmp_path, monkeypatch):
    """A FaceID whose pipeline is a stub, so no weights or camera are needed.

    Builds the real object with require_liveness=False (no model on a CI box) and
    then swaps the pipeline, so the store, the TrustedFace projection, and the
    enroll guards are all the genuine code paths.
    """

    def _make(observations, require_liveness=True):
        # Point the model lookup at a path that doesn't exist so FaceID takes its
        # no-model branch. Patching the LivenessDetector CLASS instead breaks
        # `isinstance(liveness, LivenessDetector)` inside __init__ — a stub that
        # sabotages the code it is meant to exercise, which is how the first
        # version of this fixture failed 14 tests for the wrong reason.
        import trainyourface.core.detect as detect_mod
        import trainyourface.core.embed as embed_mod
        import trainyourface.liveness.predict as predict

        monkeypatch.setattr(predict, "default_model_path", lambda: tmp_path / "absent.onnx")
        # Detector and embedder are constructed eagerly and would fetch weights.
        # The pipeline is replaced right after, so they are never used.
        monkeypatch.setattr(detect_mod, "FaceDetector", lambda *a, **k: object())
        monkeypatch.setattr(embed_mod, "FaceEmbedder", lambda *a, **k: object())

        from trainyourface.api import FaceID

        fid = FaceID(store_path=tmp_path / "store.npz", require_liveness=False)
        fid.pipeline = _StubPipeline(observations)
        fid.require_liveness = require_liveness
        return fid

    return _make


class TestFaceIDBehaviour:
    """The methods that were at 0% coverage: identify, enroll, forget, names."""

    def test_identify_projects_every_face(self, stub_faceid):
        fid = stub_faceid([_observation(name="kc", is_live=True), _observation(is_live=False)])
        faces = fid.identify(np.zeros((40, 40, 3), np.uint8))
        assert [f.status for f in faces] == ["TRUSTED", "SPOOF"]
        assert faces[0].box == (0, 0, 40, 40)

    def test_identify_on_an_empty_frame_returns_nothing(self, stub_faceid):
        fid = stub_faceid([])
        assert fid.identify(np.zeros((40, 40, 3), np.uint8)) == []

    def test_enroll_runs_the_pipeline_exactly_once(self, stub_faceid):
        """The bug this pins is a security bug, not a performance one.

        `enroll` used to call `identify()` for the liveness check and then
        `pipeline.process()` again for the embedding — a second, independent
        inference. The verdict that gated enrollment was therefore not the verdict
        belonging to the embedding actually stored, so a borderline face could
        pass the gate on run one and be saved on run two regardless.
        """
        fid = stub_faceid([_observation(is_live=True)])
        fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))
        assert fid.pipeline.calls == 1, "the check must bind to the stored artifact"

    def test_enroll_stores_the_embedding(self, stub_faceid):
        fid = stub_faceid([_observation(is_live=True)])
        face = fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))
        assert face.is_live is True
        assert fid.names == ["kc"]

    def test_enroll_refuses_a_spoof(self, stub_faceid):
        """The core security property at the enrollment boundary.

        A photo enrolled as a person makes that photo a permanent credential.
        """
        fid = stub_faceid([_observation(is_live=False, embedding=False)])
        with pytest.raises(ValueError, match="refusing to enroll"):
            fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))
        assert fid.names == [], "nothing may be stored when the check fails"

    def test_enroll_refuses_unverified_liveness(self, stub_faceid):
        """Fails closed: unknown liveness is not permission to enroll."""
        fid = stub_faceid([_observation(is_live=None)])
        with pytest.raises(ValueError, match="UNVERIFIED"):
            fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))

    def test_enroll_refuses_multiple_faces(self, stub_faceid):
        fid = stub_faceid([_observation(is_live=True), _observation(is_live=True)])
        with pytest.raises(ValueError, match="2 faces detected"):
            fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))

    def test_enroll_refuses_an_empty_frame(self, stub_faceid):
        fid = stub_faceid([])
        with pytest.raises(ValueError, match="no face detected"):
            fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))

    def test_enroll_explains_a_missing_embedding(self, stub_faceid):
        """With require_liveness=False a spoof passes the gate but is never
        embedded, because the pipeline refuses to recognize it. The error has to
        say that rather than blaming the frame."""
        fid = stub_faceid([_observation(is_live=False, embedding=False)], require_liveness=False)
        with pytest.raises(ValueError, match="nothing to enroll"):
            fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))

    def test_enroll_rejects_an_empty_name(self, stub_faceid):
        """Guarded by the store, which is the right layer — asserted here so the
        API can't later start accepting one."""
        fid = stub_faceid([_observation(is_live=True)])
        with pytest.raises(ValueError, match="cannot be empty"):
            fid.enroll("   ", np.zeros((40, 40, 3), np.uint8))

    def test_names_is_readable_and_not_a_bound_method(self, stub_faceid):
        """`store.identities` is a property, not a method.

        Calling it raised TypeError: 'list' object is not callable — and that
        shipped, because no test ever read this attribute.
        """
        fid = stub_faceid([_observation(is_live=True)])
        assert fid.names == []
        fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))
        assert fid.names == ["kc"]

    def test_forget_removes_and_persists(self, stub_faceid):
        fid = stub_faceid([_observation(is_live=True)])
        fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))
        assert fid.forget("kc") == 1
        assert fid.names == []

    def test_forget_an_unknown_name_is_zero_not_an_error(self, stub_faceid):
        fid = stub_faceid([])
        assert fid.forget("ghost") == 0

    def test_enrollment_survives_a_reload(self, stub_faceid, tmp_path):
        """Durability: enroll() saves, so a new store on the same path sees it."""
        fid = stub_faceid([_observation(is_live=True)])
        fid.enroll("kc", np.zeros((40, 40, 3), np.uint8))

        from trainyourface.core.store import EnrollmentStore

        assert EnrollmentStore(tmp_path / "store.npz").identities == ["kc"]


class TestPublicAPISurface:
    def test_importing_the_package_is_cheap(self):
        """Lazy exports keep ONNX Runtime out of a process that wanted a version."""
        import trainyourface

        assert trainyourface.__version__
        assert "LivenessDetector" in trainyourface.__all__

    def test_the_version_matches_pyproject(self):
        """Two hand-maintained copies drift.

        release.yml only checks the git tag against pyproject.toml, so a bump that
        missed __init__.py would publish a wheel whose `pip show` version and
        `__version__` disagree — permanently, since a PyPI version can never be
        re-uploaded. __version__ now reads installed metadata, and this asserts
        the two agree.

        Read with a regex rather than `tomllib`, which is stdlib only from 3.11
        while this project supports 3.10. The first version of this test imported
        tomllib and broke all three 3.10 CI jobs — a test asserting a packaging
        invariant has no business being the thing that fails on a supported
        interpreter.
        """
        import re
        from pathlib import Path

        import trainyourface

        root = Path(__file__).resolve().parent.parent
        pyproject = (root / "pyproject.toml").read_text()
        match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
        assert match, "no version line in pyproject.toml"
        assert trainyourface.__version__ == match.group(1)

    def test_the_three_public_names_resolve(self):
        from trainyourface import FaceID, LivenessDetector, TrustedFace

        assert all(callable(x) for x in (FaceID, LivenessDetector, TrustedFace))

    def test_an_unknown_attribute_still_raises(self):
        """__getattr__ must not turn typos into silent Nones."""
        import trainyourface

        with pytest.raises(AttributeError, match="no attribute"):
            _ = trainyourface.NoSuchThing

    def test_the_lazy_names_are_discoverable(self):
        """A module __getattr__ resolves names but hides them from dir().

        Without a __dir__, an IDE or REPL user can't discover the public API
        without already knowing it, which defeats the point of a public API.
        """
        import trainyourface

        listed = dir(trainyourface)
        assert {"LivenessDetector", "FaceID", "TrustedFace"} <= set(listed)


class TestTrustedFaceStates:
    """The four states, and which of them mean "let this person in"."""

    @staticmethod
    def face(name=None, is_live=None):
        from trainyourface import TrustedFace

        return TrustedFace(
            box=(0, 0, 10, 10),
            name=name,
            similarity=0.9 if name else None,
            is_live=is_live,
            spoof_score=0.01,
        )

    def test_recognized_and_live_is_trusted(self):
        f = self.face(name="kc", is_live=True)
        assert f.is_trusted is True
        assert f.status == "TRUSTED"

    def test_a_spoof_is_never_trusted_even_when_recognized(self):
        """The core security property, at the API level."""
        f = self.face(name="kc", is_live=False)
        assert f.is_trusted is False
        assert f.status == "SPOOF"

    def test_unknown_liveness_is_not_trusted(self):
        """Fails closed. The opposite choice means a photo unlocks the door."""
        f = self.face(name="kc", is_live=None)
        assert f.is_trusted is False
        assert f.status == "UNVERIFIED"

    def test_a_live_stranger_is_unknown_not_trusted(self):
        f = self.face(name=None, is_live=True)
        assert f.is_trusted is False
        assert f.status == "UNKNOWN"

    def test_it_is_immutable(self):
        """A verdict that can be reassigned downstream is not a verdict."""
        import dataclasses

        f = self.face(name="kc", is_live=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.is_live = True  # type: ignore[misc]


class TestSafeDefaults:
    def test_a_missing_model_refuses_rather_than_passing_everything(self, tmp_path):
        """A detector that can't detect would report every spoof as live."""
        from trainyourface import LivenessDetector

        with pytest.raises(FileNotFoundError, match="no liveness model"):
            LivenessDetector(model_path=tmp_path / "absent.onnx")

    def test_faceid_requires_liveness_by_default(self, tmp_path, monkeypatch):
        """`FaceID()` on a machine with no model must not silently accept photos."""
        import trainyourface.liveness.predict as predict

        monkeypatch.setattr(predict, "default_model_path", lambda: tmp_path / "absent.onnx")
        from trainyourface import FaceID

        with pytest.raises(FileNotFoundError, match="require_liveness=False"):
            FaceID(store_path=tmp_path / "s.npz")

    def test_the_error_names_the_opt_out(self, tmp_path, monkeypatch):
        """Recognition-only is legitimate, but it has to be asked for explicitly."""
        import trainyourface.liveness.predict as predict

        monkeypatch.setattr(predict, "default_model_path", lambda: tmp_path / "absent.onnx")
        from trainyourface import FaceID

        try:
            FaceID(store_path=tmp_path / "s.npz")
        except FileNotFoundError as exc:
            assert "a photo will pass" in str(exc), "the tradeoff must be stated"
