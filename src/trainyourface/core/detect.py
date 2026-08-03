"""Face detection via SCRFD on ONNX Runtime — the dlib replacement.

Why replace `dlib.get_frontal_face_detector()` at all? Two independent reasons:

1. **Install cost.** dlib compiles from source on most platforms, needing CMake
   and a C++ toolchain. That single dependency accounts for thousands of
   install-failure reports across the ecosystem. ONNX Runtime ships prebuilt
   wheels everywhere we target, including linux-aarch64 for Raspberry Pi.

2. **Accuracy.** dlib's frontal detector is a HOG cascade from 2014. It misses
   profile views, faces smaller than ~80px, and anything unevenly lit. SCRFD is a
   modern anchor-based CNN that handles all three, and it returns 5 facial
   keypoints — which we need anyway, because proper alignment before embedding is
   worth more recognition accuracy than any threshold tuning.

The old code also called `self.detector(img, 1)` with upsample=1 on every frame at
full resolution, then separately re-ran detection inside the embedding path. This
module detects once per frame at a fixed input size.
"""

from __future__ import annotations

import numpy as np

from trainyourface.core.contracts import Box
from trainyourface.core.models import DETECTOR, ModelSpec, ensure_model, resolve_providers

# SCRFD's feature pyramid strides, and 2 anchors per location per stride. These
# are properties of the trained graph, not tunables.
_STRIDES = (8, 16, 32)
_ANCHORS_PER_LOC = 2


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression.

    Written out rather than pulled from torchvision because the core install must
    not depend on torch — inference is ONNX-only by design.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        # Intersection of the top box against all remaining candidates.
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


class FaceDetector:
    """SCRFD face detector.

    Detection results carry both the box and 5 keypoints; keypoints feed
    `align.py`, which is what makes embeddings comparable across head poses.
    """

    def __init__(
        self,
        spec: ModelSpec = DETECTOR,
        *,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        prefer_gpu: bool = True,
        allow_download: bool = True,
    ) -> None:
        import onnxruntime as ort

        self.spec = spec
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        path = ensure_model(spec, allow_download=allow_download)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Log level 3 = errors only. ORT is extremely chatty about provider
        # fallbacks on stderr, which buries our own diagnostics.
        opts.log_severity_level = 3
        self.providers = resolve_providers(prefer_gpu)
        self.session = ort.InferenceSession(str(path), opts, providers=self.providers)

        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = spec.input_size

        # SCRFD variants differ in whether they emit keypoints: 6 outputs =
        # score+bbox per stride only, 9 = score+bbox+kps. Detected from the graph
        # rather than assumed, so a model swap doesn't silently misparse outputs.
        self._has_kps = len(self._output_names) == 9
        self._fmc = 3  # feature map count (one per stride)

    @property
    def active_provider(self) -> str:
        """The provider actually in use — for honest benchmark reporting."""
        return self.session.get_providers()[0]

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox to the model's input size, preserving aspect ratio.

        Aspect-preserving matters: squashing a 16:9 webcam frame into a square
        distorts faces and measurably costs both detection recall and embedding
        quality. We scale to fit and pad the remainder.
        """
        in_h, in_w = self.input_size
        h, w = image.shape[:2]
        scale = min(in_h / h, in_w / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        import cv2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # SCRFD expects RGB, normalized as (x - 127.5) / 128, NCHW.
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return blob, scale

    def detect(self, image: np.ndarray) -> tuple[list[Box], list[np.ndarray]]:
        """Detect faces in a BGR image.

        Returns (boxes, keypoints) in the ORIGINAL image's coordinate space —
        callers never deal with letterbox padding.
        """
        if image is None or image.size == 0:
            return [], []
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {image.shape}")

        blob, scale = self._preprocess(image)
        outputs = self.session.run(self._output_names, {self._input_name: blob})

        boxes_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        kps_all: list[np.ndarray] = []

        in_h, in_w = self.input_size
        for idx, stride in enumerate(_STRIDES):
            scores = outputs[idx].reshape(-1)
            # Distance-to-edge encoding: predictions are in stride units, so they
            # scale by `stride` to reach pixels.
            bbox_preds = outputs[idx + self._fmc].reshape(-1, 4) * stride

            keep = scores >= self.conf_threshold
            if not keep.any():
                continue

            # Anchor centers for this pyramid level.
            fh, fw = in_h // stride, in_w // stride
            ys, xs = np.mgrid[:fh, :fw]
            centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
            centers = np.repeat(centers.reshape(-1, 2), _ANCHORS_PER_LOC, axis=0)

            c = centers[keep]
            d = bbox_preds[keep]
            # (l, t, r, b) distances from center -> absolute corners.
            boxes = np.stack(
                [c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]],
                axis=-1,
            )
            boxes_all.append(boxes / scale)
            scores_all.append(scores[keep])

            if self._has_kps:
                kps = outputs[idx + self._fmc * 2].reshape(-1, 5, 2) * stride
                kps = kps[keep] + c[:, None, :]
                kps_all.append(kps / scale)

        if not boxes_all:
            return [], []

        boxes_np = np.vstack(boxes_all)
        scores_np = np.concatenate(scores_all)
        kps_np = np.vstack(kps_all) if kps_all else None

        keep_idx = _nms(boxes_np, scores_np, self.nms_threshold)

        h, w = image.shape[:2]
        out_boxes: list[Box] = []
        out_kps: list[np.ndarray] = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes_np[i]
            # Degenerate boxes can survive NMS; Box's validators would reject
            # them, so filter here rather than raising on a normal frame.
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            box = Box(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                score=float(np.clip(scores_np[i], 0.0, 1.0)),
            ).clipped(w, h)
            out_boxes.append(box)
            out_kps.append(kps_np[i] if kps_np is not None else np.zeros((5, 2), np.float32))

        return out_boxes, out_kps
