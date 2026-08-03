"""Liveness inference: run a trained PAD model on live frames.

This is the bridge between `train.py` (PyTorch, dev machine) and the live
pipeline (ONNX Runtime, edge device). Two backends are supported behind one
interface:

  - ONNX  -> the deployment path. No torch dependency, so a core install can run
             a trained model.
  - torch -> the dev path, for checking a fresh checkpoint before exporting it.

THE THRESHOLD IS LOADED, NOT HARDCODED
--------------------------------------
The operating point comes from `train_summary.json`, where it was chosen on the
validation split. A hardcoded 0.5 would be a different, worse threshold than the
one the reported metrics correspond to — so the number in the README and the
number the demo actually uses would silently disagree. Loading it keeps the
claim and the behavior tied together.

FAIL CLOSED
-----------
If no model is available, this module reports that fact and callers get
`liveness=None`, which `FaceObservation.is_trustworthy` treats as untrusted. It
never returns a synthetic "probably real" score: a liveness check that defaults
to pass is worse than no liveness check, because it looks like protection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from trainyourface.core.align import crop_box
from trainyourface.core.contracts import Box, LivenessResult
from trainyourface.liveness.dataset import CROP_MARGIN, INPUT_SIZE, to_model_input

# Used only when a checkpoint carries no validation threshold, which means it was
# not produced by our training loop. 0.5 is the neutral softmax midpoint.
FALLBACK_THRESHOLD = 0.5


class LivenessUnavailable(RuntimeError):
    """No usable liveness model. Raised at construction, never at predict time."""


def _softmax_last(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the final axis.

    Subtracting the max before exponentiating matters here: an INT8-quantized
    model can emit logits large enough that a naive exp overflows to inf and the
    probability comes back as nan.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_threshold(run_dir: Path | str) -> float | None:
    """Read the validation-selected threshold from a training run directory."""
    summary = Path(run_dir) / "train_summary.json"
    if not summary.exists():
        return None
    try:
        value = json.loads(summary.read_text()).get("val_threshold")
    except (json.JSONDecodeError, OSError):
        return None
    return float(value) if value is not None else None


def load_split(run_dir: Path | str) -> dict | None:
    """Read the recorded train/val/test group split from a training run directory.

    Returns None when absent — checkpoints trained before this was recorded, or a
    run directory that was moved without its summary. Callers treat that as
    "cannot verify" rather than "verified", since the two are very different
    claims to make about a held-out test set.
    """
    summary = Path(run_dir) / "train_summary.json"
    if not summary.exists():
        return None
    try:
        split = json.loads(summary.read_text()).get("split")
    except (json.JSONDecodeError, OSError):
        return None
    return split if isinstance(split, dict) and split.get("test") else None


class LivenessModel:
    """Runs PAD inference on face crops.

    Args:
        model_path: a .onnx file (deployment) or a .pt checkpoint (dev).
        threshold: overrides the stored validation threshold. Exposed because the
            right operating point is application-specific — a phone unlock wants
            a low BPCER, a door lock wants a low APCER — and the score is the
            same either way.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        threshold: float | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise LivenessUnavailable(
                f"no liveness model at {path}. Train one with `tyf train`, or run "
                "`tyf watch --no-liveness` to use recognition only (which cannot "
                "detect a printed photo)."
            )

        self.path = path
        self.input_size = INPUT_SIZE
        self._backend = "onnx" if path.suffix == ".onnx" else "torch"

        if self._backend == "onnx":
            self._init_onnx(path, prefer_gpu)
        else:
            self._init_torch(path)

        # Precedence: explicit override > threshold beside the model > fallback.
        stored = threshold if threshold is not None else load_threshold(path.parent)
        self.threshold = float(stored) if stored is not None else FALLBACK_THRESHOLD
        self.threshold_source = (
            "explicit override"
            if threshold is not None
            else "validation split (EER point)"
            if stored is not None
            else f"default {FALLBACK_THRESHOLD} (no train_summary.json found)"
        )

    def _init_onnx(self, path: Path, prefer_gpu: bool) -> None:
        import onnxruntime as ort

        from trainyourface.core.models import resolve_providers

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), opts, providers=resolve_providers(prefer_gpu)
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        # Trust the graph over our constant: an exported model fixes its input
        # resolution, and a mismatch would silently resize and shift the texture
        # statistics the model keys on.
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int):
            self.input_size = int(shape[2])

    def _init_torch(self, path: Path) -> None:
        import torch

        from trainyourface.liveness.model import ModelConfig, build_model, pick_device

        self._torch = torch
        self._device = pick_device()
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        cfg = ModelConfig(**ckpt["model_config"])
        model = build_model(cfg).to(self._device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._model = model
        self.input_size = cfg.input_size

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def active_provider(self) -> str:
        if self._backend == "onnx":
            return self.session.get_providers()[0]
        return str(self._device)

    def spoof_probability(self, crops: list[np.ndarray]) -> np.ndarray:
        """P(attack) for a batch of face crops. Crops are BGR uint8, any size."""
        import cv2

        if not crops:
            return np.zeros(0, dtype=np.float32)

        batch = np.empty((len(crops), 3, self.input_size, self.input_size), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop.shape[:2] != (self.input_size, self.input_size):
                crop = cv2.resize(
                    crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
                )
            batch[i] = to_model_input(crop)

        if self._backend == "onnx":
            logits = self.session.run([self._output_name], {self._input_name: batch})[0]
            probs = _softmax_last(np.asarray(logits, dtype=np.float32))[:, 1]
        else:
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(batch).to(self._device)
                out = self._torch.softmax(self._model(tensor), dim=1)[:, 1]
            probs = out.cpu().numpy()

        return np.clip(probs.astype(np.float32), 0.0, 1.0)

    def check(self, frame: np.ndarray, boxes: list[Box]) -> list[LivenessResult]:
        """Score every detected face in a frame.

        Crops with the same margin used in training (CROP_MARGIN) — the margin is
        load-bearing, not cosmetic. Screen bezels and paper edges sit *outside*
        the face box, so a tight crop discards the strongest replay/print cues.
        Cropping differently here than in training would also be a train/serve
        mismatch, so the constant is shared rather than duplicated.
        """
        if not boxes:
            return []

        crops = [crop_box(frame, box, size=self.input_size, margin=CROP_MARGIN) for box in boxes]

        t0 = time.perf_counter()
        probs = self.spoof_probability(crops)
        # Batch latency divided across the batch, so the number means "per face"
        # regardless of how many faces were in frame.
        per_face_ms = (time.perf_counter() - t0) * 1000.0 / len(crops)

        return [
            LivenessResult(
                spoof_probability=float(p),
                threshold=self.threshold,
                latency_ms=per_face_ms,
            )
            for p in probs
        ]


def default_model_path() -> Path:
    """Where `tyf train` / `tyf export` leave a model, in preference order.

    ONNX first: if both exist the exported model is what a user would actually
    deploy, so the demo should exercise that path rather than a checkpoint whose
    numbers might not survive export.
    """
    from trainyourface.core.models import model_dir

    candidates = [
        model_dir() / "liveness.onnx",
        Path("runs/liveness/liveness.onnx"),
        Path("runs/liveness/best.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
