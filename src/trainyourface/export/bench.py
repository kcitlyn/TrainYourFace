"""Latency benchmarking.

WHAT MAKES A LATENCY NUMBER TRUSTWORTHY
---------------------------------------
Most reported inference numbers are not reproducible, for a small set of avoidable
reasons. This module addresses each:

1. **Warmup is excluded.** The first inference pays for graph optimization,
   provider initialization and kernel compilation. Including it in a mean makes a
   fast model look slow; running only a few iterations makes the warmup dominate.

2. **Median and p95, not mean.** A mean silently absorbs the tail. On an edge
   device the tail is what actually misses a frame budget, so p95 is reported
   next to the median.

3. **The provider is named.** ONNX Runtime falls back to CPU without complaint
   when a requested provider is unavailable, so "50 ms on CoreML" is frequently
   "50 ms on CPU". The provider actually in use is printed with every result.

4. **Stages are timed separately.** An end-to-end number can't tell you whether
   the detector or the embedder is the bottleneck, which is the only thing that
   would guide an optimization.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import typer


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _report(name: str, provider: str, stats: dict[str, float]) -> None:
    typer.echo(f"\n{name}  [{provider}]")
    typer.echo(
        f"  median {stats['median']:6.2f} ms   p95 {stats['p95']:6.2f} ms   "
        f"min {stats['min']:6.2f}   max {stats['max']:6.2f}"
    )
    if stats["median"] > 0:
        typer.echo(f"  {1000.0 / stats['median']:.1f} fps at the median")


def time_callable(fn, *, runs: int, warmup: int) -> dict[str, float]:
    """Time a zero-argument callable, discarding warmup iterations."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _percentiles(samples)


def run_bench(
    *,
    model: Path | None = None,
    runs: int = 200,
    warmup: int = 20,
    full_pipeline: bool = False,
) -> dict:
    """Benchmark the liveness model, or the whole pipeline with --full-pipeline."""
    import platform

    typer.echo(f"\nhost: {platform.machine()} / {platform.system()} {platform.release()}")
    typer.echo(f"runs: {runs} timed, {warmup} warmup (excluded)")

    if full_pipeline:
        return _bench_pipeline(model, runs=runs, warmup=warmup)
    return _bench_liveness(model, runs=runs, warmup=warmup)


def _bench_liveness(model: Path | None, *, runs: int, warmup: int) -> dict:
    from trainyourface.liveness.predict import (
        LivenessModel,
        LivenessUnavailable,
        default_model_path,
    )

    path = model or default_model_path()
    try:
        liveness = LivenessModel(path)
    except LivenessUnavailable as exc:
        typer.secho(f"\n{exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    size = liveness.input_size
    results: dict[str, dict[str, float]] = {}

    # Batch 1 is the real-time case; larger batches show whether the device has
    # headroom being wasted by one-at-a-time calls.
    for batch in (1, 4, 16):
        crops = [np.random.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(batch)]
        # Larger batches take proportionally longer, so scaling the iteration
        # count keeps total benchmark time bounded.
        n = max(20, runs // batch)
        # Bind `crops` as a default arg rather than closing over the loop
        # variable: the closure is consumed immediately here, but a late-binding
        # lambda in a loop is the kind of thing that silently starts timing the
        # wrong batch size the moment this becomes lazy.
        stats = time_callable(lambda c=crops: liveness.spoof_probability(c), runs=n, warmup=warmup)
        _report(f"liveness batch={batch:<2}", liveness.active_provider, stats)
        if batch > 1:
            typer.echo(f"  {stats['median'] / batch:.2f} ms per face amortized")
        results[f"batch_{batch}"] = stats

    return results


def _bench_pipeline(model: Path | None, *, runs: int, warmup: int) -> dict:
    """Time each stage on a synthetic frame.

    A synthetic frame is used rather than a webcam feed so the benchmark is
    reproducible and runs headless in CI. The caveat is real: a frame of noise
    contains no face, so the detector finds nothing and the downstream stages
    would be skipped. So each stage is timed directly on representative input
    instead of relying on detection to feed it.
    """
    from trainyourface.cli.loader import load_pipeline
    from trainyourface.liveness.dataset import INPUT_SIZE

    pipeline = load_pipeline(use_liveness=True, quiet=True)
    typer.echo()
    for line in pipeline.describe():
        typer.echo(f"  {line}")

    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    results: dict[str, dict[str, float]] = {}

    stats = time_callable(lambda: pipeline.detector.detect(frame), runs=runs, warmup=warmup)
    _report("detect 1280x720", pipeline.detector.active_provider, stats)
    results["detect"] = stats

    if pipeline.embedder is not None:
        aligned = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        stats = time_callable(lambda: pipeline.embedder.embed([aligned]), runs=runs, warmup=warmup)
        _report("embed 1 face", pipeline.embedder.active_provider, stats)
        results["embed"] = stats

    if pipeline.liveness is not None:
        crop = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        stats = time_callable(
            lambda: pipeline.liveness.spoof_probability([crop]), runs=runs, warmup=warmup
        )
        _report("liveness 1 face", pipeline.liveness.active_provider, stats)
        results["liveness"] = stats

    total = sum(s["median"] for s in results.values())
    typer.echo(f"\none face end to end: {total:.2f} ms median -> {1000.0 / total:.1f} fps")
    if pipeline.liveness is None:
        typer.secho(
            "  NOTE: liveness not included — no model trained. The real budget is higher.",
            fg=typer.colors.YELLOW,
        )
    typer.echo()
    return results
