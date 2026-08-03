"""Tests for similarity-transform alignment.

Alignment is the fix for the train/serve preprocessing mismatch in the old
implementation (enrolled crops were dlib-aligned, live crops were not). These
tests pin the mathematical properties that make it a *similarity* transform
rather than a general affine one — shear or mirroring here would silently degrade
every embedding.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.align import (
    ARCFACE_TEMPLATE,
    CROP_SIZE,
    align_face,
    umeyama_similarity,
)


def apply(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to (N,2) points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ matrix[:2, :2].T + matrix[:, 2]


class TestUmeyama:
    def test_identity_when_already_aligned(self):
        """Template -> template must be (near) the identity transform."""
        m = umeyama_similarity(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
        assert np.allclose(m[:2, :2], np.eye(2), atol=1e-6)
        assert np.allclose(m[:, 2], 0.0, atol=1e-4)

    def test_recovers_pure_translation(self):
        src = ARCFACE_TEMPLATE
        dst = src + np.array([10.0, -5.0])
        m = umeyama_similarity(src, dst)
        assert np.allclose(apply(m, src), dst, atol=1e-4)

    def test_recovers_pure_scale(self):
        src = ARCFACE_TEMPLATE
        dst = src * 2.0
        m = umeyama_similarity(src, dst)
        assert np.allclose(apply(m, src), dst, atol=1e-4)
        # Linear part should be 2*I.
        assert np.allclose(m[:2, :2], 2.0 * np.eye(2), atol=1e-5)

    def test_recovers_rotation(self):
        theta = np.deg2rad(30.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        src = ARCFACE_TEMPLATE
        dst = src @ rot.T
        m = umeyama_similarity(src, dst)
        assert np.allclose(apply(m, src), dst, atol=1e-4)

    def test_recovers_composed_transform(self):
        theta = np.deg2rad(-17.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        src = ARCFACE_TEMPLATE
        dst = 1.7 * (src @ rot.T) + np.array([31.0, -12.0])
        m = umeyama_similarity(src, dst)
        assert np.allclose(apply(m, src), dst, atol=1e-3)

    def test_no_shear_even_when_shear_would_fit_better(self):
        """The key property: a similarity transform must refuse to shear.

        Given a sheared target, a full affine solve would match it exactly. Ours
        must NOT — the linear part has to stay a scaled rotation, because shearing
        a face to hit the template destroys the geometry the embedder reads.
        """
        shear = np.array([[1.0, 0.6], [0.0, 1.0]])
        src = ARCFACE_TEMPLATE
        dst = src @ shear.T

        m = umeyama_similarity(src, dst)
        linear = m[:2, :2]

        # A scaled rotation satisfies A^T A = s^2 I: equal column norms and
        # orthogonal columns.
        gram = linear.T @ linear
        assert gram[0, 0] == pytest.approx(gram[1, 1], rel=1e-6)
        assert gram[0, 1] == pytest.approx(0.0, abs=1e-6)
        # And it must therefore NOT reproduce the shear exactly.
        assert not np.allclose(apply(m, src), dst, atol=1.0)

    def test_no_reflection(self):
        """Mirrored input must not yield a negative determinant.

        A reflection would flip left/right, so an enrolled left-profile would
        embed like a right-profile.
        """
        src = ARCFACE_TEMPLATE
        dst = src * np.array([-1.0, 1.0])  # mirrored
        m = umeyama_similarity(src, dst)
        assert np.linalg.det(m[:2, :2]) > 0

    def test_degenerate_input_does_not_divide_by_zero(self):
        """All-identical keypoints (a failed detection) must not produce NaN."""
        src = np.zeros((5, 2), dtype=np.float32)
        m = umeyama_similarity(src, ARCFACE_TEMPLATE)
        assert np.all(np.isfinite(m))

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="matching"):
            umeyama_similarity(np.zeros((5, 2)), np.zeros((3, 2)))


class TestAlignFace:
    def test_output_shape_and_dtype(self):
        img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        kps = ARCFACE_TEMPLATE + np.array([200.0, 150.0])
        out = align_face(img, kps)
        assert out.shape == (CROP_SIZE, CROP_SIZE, 3)
        assert out.dtype == np.uint8

    def test_custom_size_scales_template(self):
        img = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        kps = ARCFACE_TEMPLATE + np.array([200.0, 150.0])
        out = align_face(img, kps, size=224)
        assert out.shape == (224, 224, 3)

    def test_landmarks_land_on_template_positions(self):
        """The whole point: after warping, keypoints must sit at template coords.

        Verified by drawing bright markers at the keypoints and checking they
        appear near the expected template locations in the output.
        """
        import cv2

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        offset = np.array([220.0, 160.0])
        # Rotate and scale the template to simulate a tilted, larger face.
        theta = np.deg2rad(20.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        kps = (ARCFACE_TEMPLATE * 1.8) @ rot.T + offset

        for x, y in kps:
            cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)

        out = align_face(img, kps)

        # Each template position should have bright pixels nearby.
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        for tx, ty in ARCFACE_TEMPLATE:
            x, y = int(round(tx)), int(round(ty))
            patch = gray[max(0, y - 5) : y + 6, max(0, x - 5) : x + 6]
            assert patch.max() > 100, f"no marker near template point ({tx:.1f}, {ty:.1f})"

    def test_wrong_keypoint_shape_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(5, 2\)"):
            align_face(img, np.zeros((3, 2), dtype=np.float32))
