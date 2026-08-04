"""The liveness (PAD) network — a small CNN trained from scratch in PyTorch.

ARCHITECTURE CHOICE, AND WHY NOT A BIG PRETRAINED BACKBONE
----------------------------------------------------------
The instinct is to fine-tune a ResNet50 or a ViT. That's the wrong call here, for
reasons specific to this task and this hardware:

1. **The cues are local texture, not semantics.** Distinguishing a real face from a
   photo of one is about moiré interference, print halftone dots, specular
   highlights, and paper flatness. That's high-frequency local information, which
   shallow convolutions capture well. ImageNet backbones are pretrained to be
   invariant to exactly this kind of low-level texture, because for object
   classification it's noise. So a big pretrained model actively discards the
   signal — a documented effect in the PAD literature, not a hunch.

2. **Deployment target.** This has to run at 30fps alongside a detector and an
   embedder on a Raspberry Pi, and train on a 4 GB-VRAM laptop GPU. A ResNet50 at
   128px doesn't fit that budget; this model is ~226K parameters at the default
   width of 32.

3. **Dataset size.** Self-collected PAD data is hundreds-to-thousands of images. A
   25M-parameter backbone on that much data memorizes subjects, which is precisely
   the failure the subject-disjoint split exists to expose.

So: a compact depthwise-separable CNN, trained from scratch. This is a real
architectural decision with a defensible rationale, which is more interesting than
`resnet50(pretrained=True)` with the head swapped.

MULTI-SCALE INPUT
-----------------
The network sees the face crop at two resolutions and fuses them. Fine texture
(moiré) lives at full resolution; global cues (a rectangular screen bezel, uniform
paper lighting) live at coarse resolution. One branch each beats a single branch
compromising between them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture hyperparameters, saved into the checkpoint.

    Persisted so a checkpoint can be rebuilt without guessing the shape it was
    trained with — a swapped width would load with mismatched keys and either
    error confusingly or, worse, partially load.
    """

    width: int = 32
    n_classes: int = 2
    dropout: float = 0.3
    input_size: int = 128


def build_model(config: ModelConfig | None = None):
    """Construct the PAD network. Imports torch lazily (train-time only dep)."""
    import torch
    import torch.nn as nn

    cfg = config or ModelConfig()

    class DepthwiseSeparable(nn.Module):
        """Depthwise conv + pointwise conv.

        ~8-9x fewer parameters and multiply-adds than a dense 3x3 conv for the
        same receptive field, which is what keeps this real-time on ARM CPU.
        """

        def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class TextureBranch(nn.Module):
        """Feature extractor for one input scale."""

        def __init__(self, width: int) -> None:
            super().__init__()
            w = width
            self.stem = nn.Sequential(
                # Stride-1 stem: downsampling immediately would throw away the
                # high-frequency texture this task depends on.
                nn.Conv2d(3, w, 3, 1, 1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                DepthwiseSeparable(w, w * 2, stride=2),
                DepthwiseSeparable(w * 2, w * 2),
                DepthwiseSeparable(w * 2, w * 4, stride=2),
                DepthwiseSeparable(w * 4, w * 4),
                DepthwiseSeparable(w * 4, w * 8, stride=2),
                DepthwiseSeparable(w * 8, w * 8),
            )
            self.out_channels = w * 8

        def forward(self, x):
            return self.blocks(self.stem(x))

    class LivenessNet(nn.Module):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__()
            self.cfg = cfg
            # Fine branch: full resolution, sees moiré and print dots.
            self.fine = TextureBranch(cfg.width)
            # Coarse branch: half resolution, sees global structure.
            self.coarse = TextureBranch(cfg.width // 2)

            fused = self.fine.out_channels + self.coarse.out_channels
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(cfg.dropout),
                nn.Linear(fused, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(128, cfg.n_classes),
            )

        def forward(self, x):
            f = self.pool(self.fine(x)).flatten(1)
            # avg_pool2d rather than interpolate: averaging is a low-pass filter,
            # so the coarse branch genuinely sees smoothed global structure rather
            # than a resampled copy of the same aliased high frequencies.
            x_coarse = torch.nn.functional.avg_pool2d(x, 2)
            c = self.pool(self.coarse(x_coarse)).flatten(1)
            return self.head(torch.cat([f, c], dim=1))

        @torch.no_grad()
        def spoof_probability(self, x):
            """Convenience: P(attack) for a batch, as used by the eval harness."""
            return torch.softmax(self.forward(x), dim=1)[:, 1]

    return LivenessNet(cfg)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, map_location=None) -> dict:
    """Load a `.pt` checkpoint without executing code from it.

    A PyTorch checkpoint is a pickle, and `torch.load(..., weights_only=False)`
    runs whatever `__reduce__` the file asks it to. This is not theoretical: I
    built a checkpoint whose load touches a file on disk, and it ran. The same
    payload with `weights_only=True` raises UnpicklingError instead.

    That matters here specifically because sharing checkpoints is a workflow this
    project encourages — `tyf eval --checkpoint` pointed at someone else's model is
    the documented way to reproduce a reported number, and PAD models are exactly
    the kind of artifact people pass around. Opening one should not be equivalent
    to running an unknown script.

    Nothing is given up by restricting it: these checkpoints hold tensors, plain
    dicts, ints and floats, all of which `weights_only=True` allows. If a future
    checkpoint needs a custom class, the fix is `torch.serialization`'s explicit
    allowlist for that class, not turning the guard off globally.
    """
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - torch raises several types here
        # Do NOT fall back to weights_only=False. A checkpoint that fails to load
        # safely is either from an older/odd producer or is hostile, and this code
        # cannot tell those apart — so it reports rather than guesses, and names
        # the escape hatch instead of taking it silently.
        raise ValueError(
            f"could not safely load {path}: {exc}\n"
            "This checkpoint contains objects beyond tensors and plain data. It is "
            "loaded with weights_only=True because a PyTorch checkpoint is a pickle "
            "and can execute code on load. If you produced this file yourself and "
            "trust it, re-save it with `torch.save(ckpt, path)` from a current "
            "version, or allowlist the specific class via "
            "torch.serialization.add_safe_globals()."
        ) from exc


def pick_device(prefer: str | None = None):
    """Choose a training device.

    Order is deliberate for this project's hardware: CUDA (the RTX 500 Ada
    laptop), then MPS (Apple Silicon — usable for training a model this small),
    then CPU. Reported explicitly rather than silently chosen so benchmark numbers
    always name the device they came from.
    """
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
