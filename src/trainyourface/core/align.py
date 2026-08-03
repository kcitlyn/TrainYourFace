"""Similarity-transform face alignment from 5 keypoints.

This module is small but it is one of the two biggest accuracy wins over the old
implementation, so it's worth stating why.

The old code embedded faces two different ways depending on the code path:
  - `train_face_images()` called `dlib.get_face_chip()`, which DOES align.
  - `get_face_descriptor()` (the live path) called
    `compute_face_descriptor(img, shape)` on the raw frame — no alignment.

So enrolled embeddings were computed on aligned crops while live embeddings were
computed on unaligned ones. Those two live in slightly different regions of
embedding space, which inflates distances between the same person and forces the
threshold down to compensate. That is very likely why the original needed a
threshold of 0.4 (unusually strict for dlib, where 0.6 is the standard value) and
why the README told users to "take more photos of your face" when recognition
failed — the real problem was a train/serve preprocessing mismatch, not sample
count.

Fixing it means one alignment function, used by both paths. ArcFace models are
trained on faces warped to a canonical 112x112 layout, so matching that layout at
inference is what the weights expect.
"""

from __future__ import annotations

import numpy as np

# The canonical 5-point layout ArcFace/insightface models are trained against,
# in 112x112 pixel coordinates: left eye, right eye, nose tip, left mouth
# corner, right mouth corner. These constants come from the insightface
# reference implementation and must not be "tidied" — they define the target
# space the weights were fit to.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

CROP_SIZE = 112


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation + uniform scale + translation).

    Returns a 2x3 matrix suitable for `cv2.warpAffine`.

    We use a similarity transform rather than a full affine on purpose: affine
    allows shear and non-uniform scaling, which can stretch a face to hit the
    template exactly and in doing so destroy the geometry the embedder relies on.
    Similarity can only rotate, scale, and translate — it normalizes pose without
    editing the face.

    Implements the Umeyama (1991) closed-form solution.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"expected matching (N,2) point sets, got {src.shape} and {dst.shape}")

    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Cross-covariance, then SVD.
    cov = dst_demean.T @ src_demean / n
    u, s, vt = np.linalg.svd(cov)

    # Guard against a reflection: a naive U @ Vt can produce det < 0, which
    # mirrors the face. Flipping the sign of the smallest singular direction is
    # the standard correction.
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0

    rotation = u @ np.diag(d) @ vt

    src_var = src_demean.var(axis=0).sum()
    # Degenerate input (all keypoints identical) would divide by zero.
    scale = 1.0 if src_var < 1e-12 else (d * s).sum() / src_var

    translation = dst_mean - scale * rotation @ src_mean

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:2, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix


def align_face(
    image: np.ndarray,
    keypoints: np.ndarray,
    size: int = CROP_SIZE,
) -> np.ndarray:
    """Warp a face to the canonical ArcFace layout.

    Args:
        image: full BGR frame.
        keypoints: (5, 2) detected landmarks in image coordinates.
        size: output edge length; the template scales linearly with it.

    Returns a (size, size, 3) BGR crop.
    """
    import cv2

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (5, 2):
        raise ValueError(f"expected (5, 2) keypoints, got {keypoints.shape}")
    # Checked here rather than letting the SVD fail: non-finite landmarks surface
    # as "LinAlgError: SVD did not converge" from inside umeyama_similarity, which
    # names neither the keypoints nor the frame that produced them.
    if not np.all(np.isfinite(keypoints)):
        raise ValueError("keypoints contain NaN or inf; cannot compute an alignment")

    template = ARCFACE_TEMPLATE * (size / CROP_SIZE)
    matrix = umeyama_similarity(keypoints, template)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)


def crop_box(image: np.ndarray, box, size: int = CROP_SIZE, margin: float = 0.0) -> np.ndarray:
    """Plain box crop — the fallback when keypoints are unavailable.

    Used by the liveness model, which wants the region *around* the face
    (including some background) rather than a tightly aligned crop: screen bezels,
    paper edges, and moiré patterns are exactly the cues that reveal a replay or
    print attack, and aggressive alignment crops them away.

    Coordinates are clamped to the image on BOTH paths. This is not defensive
    boilerplate: detectors routinely emit boxes running off the frame edge, and a
    negative x1 makes `image[y1:y2, x1:x2]` a *negative index* — numpy reads from
    the far side of the array and the slice comes back empty, so the function
    returned an all-black crop for any face touching the left or top edge. The
    liveness model then scored a black square instead of the face, and nothing
    errored. The margin path happened to clamp already; the margin-free path (used
    by the recognition fallback) did not.
    """
    import cv2

    h, w = image.shape[:2]
    dx, dy = (int(box.width * margin), int(box.height * margin)) if margin else (0, 0)
    x1, y1 = max(0, box.x1 - dx), max(0, box.y1 - dy)
    x2, y2 = min(w, box.x2 + dx), min(h, box.y2 + dy)

    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((size, size, 3), dtype=image.dtype)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
