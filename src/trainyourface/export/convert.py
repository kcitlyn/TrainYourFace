"""Export a trained checkpoint for edge deployment.

WHY EXPORT IS VERIFIED BY DEFAULT
---------------------------------
An export can succeed and still be wrong. Ops get fused, a dynamic axis gets
frozen to 1, BatchNorm folds slightly differently, a quantized layer saturates.
None of that raises — you get a model that loads, runs, and is subtly miscalibrated,
which for a PAD model means the threshold chosen during training no longer
corresponds to the same operating point. The reported ACER then describes a model
nobody is running.

So `export_model` compares PyTorch and exported outputs on random inputs and
reports the maximum divergence. Verification is on by default and has to be
switched off explicitly.

ON INT8
-------
Quantization is offered because it's the interesting edge tradeoff, but it is
reported honestly: this model is ~0.9 MB in fp32, so INT8's value here is
throughput on integer-only hardware, not size. And PAD depends on fine texture
statistics, which is exactly the kind of signal quantization can blunt — so the
correct claim is only ever made after re-running `tyf eval` on the quantized
model, never inferred from the fp32 numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

# Divergence above this means the exported graph is not computing the same
# function. Chosen for fp32 export, where agreement should be near-exact;
# quantized models are checked against a much looser bound.
FP32_TOLERANCE = 1e-4
INT8_TOLERANCE = 0.05


def export_model(
    checkpoint: Path | str,
    *,
    out: Path | None = None,
    fmt: str = "onnx",
    quantize: bool = False,
    verify: bool = True,
) -> Path:
    """Export a .pt checkpoint. Returns the written path."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise typer.BadParameter(f"no checkpoint at {checkpoint}")

    fmt = fmt.lower()
    if fmt not in ("onnx", "coreml"):
        raise typer.BadParameter(f"unknown format {fmt!r}; use onnx or coreml")
    if quantize and fmt != "onnx":
        raise typer.BadParameter("--quantize applies to ONNX export only")

    import torch

    from trainyourface.liveness.model import ModelConfig, build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    size = cfg.input_size
    example = torch.randn(1, 3, size, size)

    if fmt == "onnx":
        path = out or checkpoint.parent / "liveness.onnx"
        _export_onnx(model, example, Path(path))
        if quantize:
            path = _quantize_onnx(Path(path))
        if verify:
            _verify_onnx(model, Path(path), size, quantized=quantize)
    else:
        path = out or checkpoint.parent / "liveness.mlpackage"
        _export_coreml(model, example, Path(path))
        if verify:
            _verify_coreml(model, Path(path), size)

    path = Path(path)
    mb = _size_mb(path)
    typer.secho(f"\nexported {path}  ({mb:.2f} MB)", fg=typer.colors.GREEN)

    # The threshold must travel with the model. Without it the deployed model
    # falls back to 0.5, which is a different operating point than the one the
    # reported metrics describe.
    summary = checkpoint.parent / "train_summary.json"
    if summary.exists() and path.parent != checkpoint.parent:
        import shutil

        shutil.copy2(summary, path.parent / "train_summary.json")
        typer.echo("  copied train_summary.json (carries the validation threshold)")

    if quantize:
        typer.secho(
            "  INT8: re-run `tyf eval` against this model before quoting its metrics. "
            "Quantization can blunt the fine texture cues PAD depends on, and the "
            "fp32 numbers do not transfer.",
            fg=typer.colors.YELLOW,
        )
    return path


def _export_onnx(model, example, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=["input"],
            output_names=["logits"],
            # Batch stays dynamic so the same file serves one face per frame and
            # batched offline evaluation. A frozen batch of 1 would make eval
            # needlessly slow, and is an easy thing to not notice.
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    except ModuleNotFoundError as exc:
        # torch >=2.6 exports through a dynamo path that needs onnxscript. The
        # native failure is a bare ModuleNotFoundError raised several frames deep
        # inside torch, which looks like a torch bug rather than a missing extra.
        raise RuntimeError(
            f"ONNX export needs an extra package that isn't installed ({exc.name}).\n"
            "  pip install 'trainyourface[export]'"
        ) from exc


def _quantize_onnx(path: Path) -> Path:
    """Dynamic INT8 quantization. Weights to int8, activations quantized at runtime.

    Dynamic rather than static because static requires a calibration dataset, and
    calibrating on training data then reporting test metrics is a subtle leak.
    Dynamic needs no calibration data at all.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = path.with_name(path.stem + "_int8.onnx")
    quantize_dynamic(str(path), str(out), weight_type=QuantType.QInt8)
    typer.echo(f"  quantized: {_size_mb(path):.2f} MB -> {_size_mb(out):.2f} MB")
    return out


def _verify_onnx(model, path: Path, size: int, *, quantized: bool) -> None:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    # Several random batches, including a batch > 1, which is what catches a
    # dynamic axis that got frozen during export.
    worst = 0.0
    for batch in (1, 1, 4):
        x = np.random.randn(batch, 3, size, size).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {name: x})[0]
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"export shape mismatch at batch={batch}: onnx {actual.shape} vs "
                f"torch {expected.shape}. A dynamic axis was likely frozen."
            )
        worst = max(worst, float(np.abs(actual - expected).max()))

    tolerance = INT8_TOLERANCE if quantized else FP32_TOLERANCE
    typer.echo(f"  verified: max |onnx - torch| = {worst:.2e} (tolerance {tolerance:.0e})")
    if worst > tolerance:
        raise RuntimeError(
            f"exported model diverges from PyTorch by {worst:.2e}, above the "
            f"{tolerance:.0e} tolerance. Do not deploy this: the training-time "
            "threshold would not correspond to the same operating point."
        )


def _export_coreml(model, example, path: Path) -> None:
    """Convert to Core ML for the Apple Neural Engine.

    ML_PROGRAM (mlpackage) rather than the older NeuralNetwork format because it's
    the one that can actually target the ANE on Apple Silicon.
    """
    import coremltools as ct
    import torch

    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(str(path))


def _verify_coreml(model, path: Path, size: int) -> None:
    import coremltools as ct
    import torch

    mlmodel = ct.models.MLModel(str(path))
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    out = mlmodel.predict({"input": x})
    actual = np.asarray(next(iter(out.values())), dtype=np.float32).reshape(expected.shape)

    worst = float(np.abs(actual - expected).max())
    # Core ML runs fp16 on the ANE by design, so exact fp32 agreement is not the
    # expectation — a loose bound here is correct, not a lowered standard.
    typer.echo(f"  verified: max |coreml - torch| = {worst:.2e} (fp16 compute, loose bound)")
    if worst > 0.05:
        raise RuntimeError(f"Core ML output diverges by {worst:.2e}; do not deploy")


def _size_mb(path: Path) -> float:
    if path.is_dir():  # .mlpackage is a directory
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6
