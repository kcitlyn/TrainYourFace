"""Face embedding via ArcFace on ONNX Runtime.

Replaces `dlib.face_recognition_model_v1`. Beyond dropping the dlib dependency,
the substantive changes from the old implementation:

- **512-D ArcFace instead of 128-D dlib ResNet.** ArcFace's angular-margin
  training produces embeddings where cosine similarity is directly meaningful,
  and it benchmarks far better on LFW/CFP-FP/AgeDB than the 2017-era dlib net.

- **L2-normalized output, compared by cosine similarity.** The old code used raw
  Euclidean distance on unnormalized descriptors, where vector magnitude — which
  carries no identity information, only image contrast and lighting — influenced
  the distance. Normalizing first makes comparisons depend on direction only.

- **Batched.** The old code embedded one face per call inside a per-face loop
  inside a per-frame loop. Batching multiple faces into one session run matters
  once several people are in frame.
"""

from __future__ import annotations

import numpy as np

from trainyourface.core.contracts import EMBEDDING_DIM
from trainyourface.core.models import EMBEDDER, ModelSpec, ensure_model, resolve_providers


def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """Scale vectors to unit length.

    After this, cosine similarity is a plain dot product, and it equals
    1 - (euclidean^2 / 2) — so the two metrics rank identically and we can use
    the more interpretable one. Cosine is bounded in [-1, 1], which makes a
    threshold portable across models; raw Euclidean distance is not.
    """
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


class FaceEmbedder:
    """ArcFace embedding model."""

    def __init__(
        self,
        spec: ModelSpec = EMBEDDER,
        *,
        prefer_gpu: bool = True,
        allow_download: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.spec = spec
        path = ensure_model(spec, allow_download=allow_download)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self.providers = resolve_providers(prefer_gpu)
        self.session = ort.InferenceSession(str(path), opts, providers=self.providers)

        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self.input_size = spec.input_size

    @property
    def active_provider(self) -> str:
        return self.session.get_providers()[0]

    def embed(self, aligned_faces: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Embed one or more aligned 112x112 BGR crops.

        Args:
            aligned_faces: a single (112,112,3) crop or a list of them. These must
                come from `align.align_face` — passing an unaligned crop is the
                exact train/serve mismatch bug this rewrite fixes.

        Returns an (N, 512) array of L2-normalized float32 embeddings.
        """
        import cv2

        if isinstance(aligned_faces, np.ndarray) and aligned_faces.ndim == 3:
            aligned_faces = [aligned_faces]
        if len(aligned_faces) == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        h, w = self.input_size
        batch = np.empty((len(aligned_faces), 3, h, w), dtype=np.float32)
        for i, face in enumerate(aligned_faces):
            if face.shape[:2] != (h, w):
                face = cv2.resize(face, (w, h), interpolation=cv2.INTER_LINEAR)
            # ArcFace expects RGB scaled to [-1, 1] as (x - 127.5) / 127.5.
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
            batch[i] = np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))

        raw = self.session.run([self._output_name], {self._input_name: batch})[0]
        raw = np.asarray(raw, dtype=np.float32).reshape(len(aligned_faces), -1)

        if raw.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"model {self.spec.name} emitted {raw.shape[1]}-D embeddings, "
                f"expected {EMBEDDING_DIM}-D. Check the model spec."
            )
        return l2_normalize(raw, axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between L2-normalized embeddings.

    Assumes inputs are already normalized (a dot product then). Accepts (D,) or
    (N,D) for either argument and broadcasts to (N,M).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return a @ b.T
