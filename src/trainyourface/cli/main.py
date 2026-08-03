"""The `tyf` command line interface.

Replaces a `input("1 for training; 2 for identification")` menu, which meant the
tool could not be scripted, tested, or run on a headless device. Every stage of
the project is now a subcommand, and the ones that matter for reproducing a
reported number (`train`, `eval`, `bench`) run without a display.

Command groups:
    watch / enroll / list / forget   -- use it
    capture / manifest               -- build a PAD dataset
    train / eval / calibrate         -- produce and defend a model
    export / bench                   -- ship it to an edge device
    info                             -- what's actually loaded, and where
"""

from __future__ import annotations

from pathlib import Path

import typer

from trainyourface.core.contracts import AttackType

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Offline face recognition that knows when it's being fooled.",
    rich_markup_mode=None,
)

DEFAULT_RUN_DIR = Path("runs/liveness")
DEFAULT_DATA_DIR = Path("data/pad")


# --------------------------------------------------------------------------
# use it
# --------------------------------------------------------------------------


@app.command()
def watch(
    camera: int = typer.Option(0, help="Camera device index."),
    liveness: Path | None = typer.Option(None, help="Path to a liveness .onnx or .pt."),
    no_liveness: bool = typer.Option(
        False, "--no-liveness", help="Run recognition only. A photo will pass."
    ),
    liveness_threshold: float | None = typer.Option(
        None, help="Override the validation-selected spoof threshold."
    ),
    match_threshold: float | None = typer.Option(None, help="Override the recognition threshold."),
    store: Path | None = typer.Option(None, help="Enrollment store path."),
    identify_spoofs: bool = typer.Option(
        False,
        "--identify-spoofs",
        help="Also identify faces that failed liveness. For evaluation only: it "
        "confirms to an attacker that their spoof targeted the right person.",
    ),
) -> None:
    """Live camera view with liveness + recognition overlay."""
    from trainyourface.cli.live import run_watch

    run_watch(
        camera=camera,
        liveness_path=liveness,
        use_liveness=not no_liveness,
        liveness_threshold=liveness_threshold,
        match_threshold=match_threshold,
        store_path=store,
        identify_spoofs=identify_spoofs,
    )


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Who this is."),
    camera: int = typer.Option(0, help="Camera device index."),
    samples: int = typer.Option(8, min=1, max=40, help="Frames to collect."),
    store: Path | None = typer.Option(None, help="Enrollment store path."),
    relationship: str | None = typer.Option(None, help="Free-form note, e.g. 'family'."),
    require_live: bool = typer.Option(
        True,
        "--require-live/--no-require-live",
        help="Refuse to enroll a face that fails liveness. Disabling this allows "
        "enrolling from a photo, which permanently installs a spoof as ground truth.",
    ),
    liveness: Path | None = typer.Option(None, help="Path to a liveness model."),
) -> None:
    """Enroll a person from the webcam."""
    from trainyourface.cli.live import run_enroll

    run_enroll(
        name,
        camera=camera,
        samples=samples,
        store_path=store,
        relationship=relationship,
        require_live=require_live,
        liveness_path=liveness,
    )


@app.command(name="list")
def list_identities(
    store: Path | None = typer.Option(None, help="Enrollment store path."),
) -> None:
    """List enrolled identities."""
    from trainyourface.core.store import EnrollmentStore

    s = EnrollmentStore(store)
    s.load()

    if s.is_empty:
        typer.echo(f"no identities enrolled ({s.path})")
        return

    typer.echo(f"\n{len(s.identities)} identities, {len(s)} embeddings  [{s.path}]\n")
    typer.echo(f"  {'name':<20} {'samples':>7}  {'relationship':<14} enrolled")
    for row in s.summary():
        typer.echo(
            f"  {row['name']:<20} {row['samples']:>7}  "
            f"{row['relationship'] or '-':<14} {row['enrolled_at'] or '-'}"
        )
    typer.echo()


@app.command()
def forget(
    name: str = typer.Argument(..., help="Identity to delete."),
    store: Path | None = typer.Option(None, help="Enrollment store path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete an identity and all of its embeddings.

    Face templates are biometric data, so erasure has to be a first-class
    operation rather than something you do by hand-editing a file.
    """
    from trainyourface.core.store import EnrollmentStore

    s = EnrollmentStore(store)
    s.load()

    count = s.count_for(name)
    if count == 0:
        typer.secho(f"{name!r} is not enrolled", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    if not yes:
        typer.confirm(f"Delete {name!r} and its {count} embeddings?", abort=True)

    removed = s.remove(name)
    s.save()
    typer.secho(f"removed {name!r} ({removed} embeddings)", fg=typer.colors.GREEN)


# --------------------------------------------------------------------------
# build a dataset
# --------------------------------------------------------------------------


@app.command()
def capture(
    label: AttackType = typer.Option(..., "--label", help="What is in front of the camera."),
    subject: str = typer.Option(..., "--subject", help="Who — the unit of dataset splitting."),
    session: str = typer.Option("s1", help="Capture session id. Same person, different day."),
    target: int = typer.Option(120, min=1, help="Frames to collect."),
    instrument: str | None = typer.Option(
        None, help="Attack device, e.g. 'iphone13' or 'laser_matte'. Reported per-type in eval."
    ),
    out: Path = typer.Option(DEFAULT_DATA_DIR, help="Dataset directory."),
    camera: int = typer.Option(0, help="Camera device index."),
) -> None:
    """Record labeled bona-fide or attack frames for PAD training.

    The academic PAD datasets all require signed licenses and can't be shipped,
    so the dataset is yours. Capture at least 3 subjects — fewer makes a
    subject-disjoint split impossible and `train` will refuse.
    """
    from trainyourface.cli.capture import run_capture

    run_capture(
        out_dir=out,
        label=label,
        subject=subject,
        session=session,
        target=target,
        instrument=instrument,
        camera=camera,
    )


@app.command()
def manifest(
    data: Path = typer.Option(DEFAULT_DATA_DIR, help="Dataset directory."),
    check: bool = typer.Option(True, help="Verify every listed image exists."),
) -> None:
    """Summarize a PAD dataset and check it's ready to train on."""
    from trainyourface.liveness.dataset import DatasetManifest, subject_disjoint_split

    path = data / "manifest.json"
    if not path.exists():
        typer.secho(f"no manifest at {path}. Run `tyf capture` first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    m = DatasetManifest.load(path)
    typer.echo(f"\n{m.describe()}\n")

    if check:
        image_root = m.image_root(data)
        missing = [s.path for s in m.samples if not (image_root / s.path).exists()]
        if missing:
            typer.secho(f"{len(missing)} listed images are missing:", fg=typer.colors.RED)
            for p in missing[:5]:
                typer.echo(f"  {p}")
            raise typer.Exit(code=1)
        typer.secho(f"all {len(m.samples)} images present", fg=typer.colors.GREEN)

    # Report the split now rather than at train time — finding out you need more
    # subjects after setting up a training run is worse than finding out here.
    try:
        train_s, val_s, test_s = subject_disjoint_split(m.samples)
        typer.echo(
            f"\nsubject-disjoint split: {len(train_s)} train / {len(val_s)} val / "
            f"{len(test_s)} test"
        )
    except ValueError as exc:
        typer.secho(f"\nnot splittable yet: {exc}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1) from exc


@app.command(name="import-celeba")
def import_celeba(
    root: Path = typer.Argument(..., help="CelebA-Spoof directory (the one containing Data/)."),
    out: Path = typer.Option(DEFAULT_DATA_DIR, help="Where to write manifest.json."),
    label_file: Path = typer.Option(None, help="Annotation JSON. Default: metas/intra_test/."),
    limit: int = typer.Option(
        None, help="Cap the sample count. Samples whole subjects, not random images."
    ),
) -> None:
    """Convert a downloaded CelebA-Spoof dataset into a manifest.

    625K images, 10,177 subjects, direct download with no license application —
    the only public PAD dataset that doesn't need a signed institutional
    agreement. Non-commercial research use only, per the dataset's terms; nothing
    from it is redistributed here.
    """
    from trainyourface.liveness.celeba_spoof import convert

    try:
        m = convert(root, label_file=label_file, limit=limit, log=typer.echo)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"import failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    # Point the manifest's root at the CelebA tree so images resolve from wherever
    # the manifest lands. Copying 625K images to sit beside it is not an option.
    out.mkdir(parents=True, exist_ok=True)
    m.root = str(Path(root).resolve())
    m.save(out / "manifest.json")

    typer.echo(f"\n{m.describe()}\n")
    typer.secho(f"wrote {out / 'manifest.json'}", fg=typer.colors.GREEN)
    typer.echo(
        "\nCite: Zhang et al., CelebA-Spoof: Large-Scale Face Anti-Spoofing "
        "Dataset with Rich Annotations, ECCV 2020."
    )


# --------------------------------------------------------------------------
# train and defend a model
# --------------------------------------------------------------------------


@app.command()
def train(
    data: Path = typer.Option(DEFAULT_DATA_DIR, help="Dataset directory with manifest.json."),
    out: Path = typer.Option(DEFAULT_RUN_DIR, help="Where to write checkpoints."),
    epochs: int = typer.Option(30, min=1),
    batch_size: int = typer.Option(64, min=2),
    learning_rate: float = typer.Option(3e-4),
    width: int = typer.Option(32, help="Model width. Smaller = faster on device."),
    seed: int = typer.Option(0),
    split_by: str = typer.Option(
        "subject", help="'subject' or 'session'. Session is stricter.", metavar="MODE"
    ),
    device: str | None = typer.Option(None, help="cuda / mps / cpu. Auto-detected by default."),
) -> None:
    """Train the liveness model, then evaluate it on a held-out test split.

    The split is subject-disjoint and integrity-checked; the threshold is chosen
    on validation and applied unchanged to test. Both are enforced here rather
    than left to the caller, because either one done wrong turns the reported
    numbers into noise that still looks good.
    """
    from trainyourface.eval.report import evaluate_checkpoint, render_report, save_report
    from trainyourface.liveness.dataset import (
        DatasetManifest,
        check_split_integrity,
        split_fingerprint,
        subject_disjoint_split,
    )
    from trainyourface.liveness.model import ModelConfig
    from trainyourface.liveness.train import TrainConfig
    from trainyourface.liveness.train import train as run_train

    manifest_path = data / "manifest.json"
    if not manifest_path.exists():
        typer.secho(
            f"no manifest at {manifest_path}. Run `tyf capture` first.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    m = DatasetManifest.load(manifest_path)
    typer.echo(f"\n{m.describe()}\n")

    try:
        train_s, val_s, test_s = subject_disjoint_split(m.samples, seed=seed, by=split_by)
        check_split_integrity(train_s, val_s, test_s, by=split_by)
    except ValueError as exc:
        typer.secho(f"split failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"split by {split_by}: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test "
        "(disjoint, verified)",
        fg=typer.colors.CYAN,
    )

    cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        model=ModelConfig(width=width),
    )
    # The manifest knows where its images live; --data is only the fallback. An
    # imported manifest points at the source dataset tree.
    image_root = m.image_root(data)
    summary = run_train(
        train_s,
        val_s,
        root=image_root,
        out_dir=out,
        config=cfg,
        # Recorded so `tyf eval` can verify it re-derived the same partition
        # instead of trusting that whoever ran it passed the same flags.
        split=split_fingerprint(train_s, val_s, test_s, by=split_by),
    )

    # Test runs once, at the val-selected threshold. Iterating on this number is
    # how a test split stops being held out.
    typer.secho("\nevaluating on the held-out test split...", fg=typer.colors.CYAN)
    result = evaluate_checkpoint(
        out / "best.pt",
        test_s,
        root=image_root,
        threshold=summary["val_threshold"],
        device=device,
    )
    typer.echo(render_report(result, title="Test Set (subject-disjoint)"))
    save_report(result, out / "test_report.json")
    typer.echo(f"report: {out / 'test_report.json'}")


@app.command(name="eval")
def eval_cmd(
    checkpoint: Path = typer.Option(DEFAULT_RUN_DIR / "best.pt", help="Model to evaluate."),
    data: Path = typer.Option(DEFAULT_DATA_DIR, help="Dataset directory."),
    threshold: float | None = typer.Option(
        None, help="Defaults to the validation threshold from train_summary.json."
    ),
    seed: int = typer.Option(0, help="Must match the training seed to get the same split."),
    split_by: str = typer.Option("subject", metavar="MODE"),
    markdown: bool = typer.Option(False, help="Emit a markdown table for the README."),
    cross: Path = typer.Option(
        None,
        help="A SECOND dataset dir to test on. Reports the in-domain vs out-of-domain gap.",
    ),
) -> None:
    """Evaluate a checkpoint on the held-out test split.

    The threshold defaults to the one chosen on validation during training. Pass
    --threshold only to explore a different operating point, and label it as such
    if you report it.

    With --cross, also evaluates on a dataset the model never trained on and
    reports the gap. That gap is the number worth publishing: every PAD model
    scores well in-domain, and the drop on unseen cameras and rooms is what says
    whether it learned the phenomenon or just the dataset.
    """
    from trainyourface.eval.report import (
        cross_dataset_report,
        evaluate_checkpoint,
        render_markdown_table,
        render_report,
    )
    from trainyourface.liveness.dataset import (
        DatasetManifest,
        check_split_integrity,
        subject_disjoint_split,
        verify_split_matches,
    )
    from trainyourface.liveness.predict import load_split, load_threshold

    if not checkpoint.exists():
        typer.secho(f"no checkpoint at {checkpoint}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    thr = threshold if threshold is not None else load_threshold(checkpoint.parent)
    if thr is None:
        typer.secho(
            f"no threshold given and no train_summary.json beside {checkpoint}. "
            "Pass --threshold explicitly, and label it as hand-chosen when reporting.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # The same guards as `train`. `eval` is the command run against someone
    # else's checkpoint or a moved dataset, so it is the one most likely to be
    # pointed at a missing manifest — and a raw FileNotFoundError traceback names
    # a path without saying which flag to change.
    manifest_path = data / "manifest.json"
    if not manifest_path.exists():
        typer.secho(
            f"no manifest at {manifest_path}. Pass --data pointing at the dataset "
            "the checkpoint was trained on.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    m = DatasetManifest.load(manifest_path)
    try:
        train_s, val_s, test_s = subject_disjoint_split(m.samples, seed=seed, by=split_by)
        check_split_integrity(train_s, val_s, test_s, by=split_by)
    except ValueError as exc:
        typer.secho(f"split failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    # `--seed` and `--split-by` default to 0/"subject" no matter what training
    # used, so a mismatch silently re-partitions the data and puts trained-on
    # subjects into the "held-out" test set. The result still looks correct —
    # internally disjoint, normal-looking report — so it is checked, not assumed.
    fingerprint = load_split(checkpoint.parent)
    if fingerprint:
        try:
            verify_split_matches(fingerprint, test_s, by=split_by)
        except ValueError as exc:
            typer.secho(f"\nSPLIT MISMATCH: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
        typer.secho(
            f"test split matches training ({len(fingerprint['test'])} held-out "
            f"{fingerprint['by']}s, verified)",
            fg=typer.colors.CYAN,
        )
    else:
        typer.secho(
            "no split recorded beside this checkpoint; cannot verify the test set "
            "is the one held out during training.",
            fg=typer.colors.YELLOW,
        )

    result = evaluate_checkpoint(checkpoint, test_s, root=m.image_root(data), threshold=thr)
    if markdown:
        typer.echo(render_markdown_table(result))
    else:
        typer.echo(render_report(result, title="Test Set (subject-disjoint)"))

    if cross is None:
        return

    # Cross-dataset: EVERY sample of the second dataset is test data, with no
    # split at all. There is nothing to hold out — the model never saw any of it,
    # which is the entire point. Splitting here would throw away most of the
    # evaluation set for no gain.
    cross_manifest = cross / "manifest.json"
    if not cross_manifest.exists():
        typer.secho(
            f"no manifest at {cross_manifest}. --cross wants a second dataset "
            "directory, e.g. your own captures when the model trained on CelebA.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    cross_m = DatasetManifest.load(cross_manifest)
    if not cross_m.samples:
        typer.secho(f"{cross_manifest} lists no samples", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # An overlapping subject would make "out-of-domain" false in the same way the
    # split-reconstruction bug made "held-out" false, so it is checked here too.
    # Different datasets can genuinely reuse subject IDs — CelebA numbers its
    # identities from 1, and so might a hand-built manifest.
    shared = {s.subject for s in train_s} & {s.subject for s in cross_m.samples}
    if shared:
        typer.secho(
            f"\nWARNING: {len(shared)} subject id(s) appear in BOTH the training "
            f"data and --cross: {sorted(shared)[:5]}. If those are the same people, "
            "this is not an out-of-domain test. If the two datasets just number "
            "their subjects the same way, rename one side to make it verifiable.",
            fg=typer.colors.YELLOW,
        )

    cross_result = evaluate_checkpoint(
        checkpoint, cross_m.samples, root=cross_m.image_root(cross), threshold=thr
    )
    typer.echo(render_report(cross_result, title=f"Cross-dataset: {cross.name} (never trained on)"))
    typer.echo(
        cross_dataset_report(
            result, cross_result, source_name=data.name or "train set", target_name=cross.name
        )
    )


@app.command()
def calibrate(
    store: Path | None = typer.Option(None, help="Enrollment store path."),
) -> None:
    """Pick a recognition threshold from your own enrollments.

    The default 0.36 comes from a benchmark, not from your camera, your lighting,
    or the people you enrolled. This computes genuine and impostor similarity
    distributions from the store and reports where they actually separate.
    """
    import numpy as np

    from trainyourface.core.store import EnrollmentStore

    s = EnrollmentStore(store)
    s.load()
    genuine, impostor = s.genuine_impostor_scores()

    if len(genuine) == 0 or len(impostor) == 0:
        typer.secho(
            "need at least 2 identities with 2+ samples each to calibrate. "
            f"Currently: {len(s.identities)} identities, {len(s)} embeddings.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"\ngenuine pairs   {len(genuine):5d}  mean {genuine.mean():.4f}  min {genuine.min():.4f}"
    )
    typer.echo(
        f"impostor pairs  {len(impostor):5d}  mean {impostor.mean():.4f}  max {impostor.max():.4f}"
    )

    gap = genuine.min() - impostor.max()
    if gap > 0:
        # Cleanly separable: any threshold in the gap gives zero errors on this
        # data. The midpoint maximizes margin on both sides.
        suggested = (genuine.min() + impostor.max()) / 2
        typer.secho(
            f"\nfully separable, margin {gap:.4f}\n  suggested threshold: {suggested:.4f}",
            fg=typer.colors.GREEN,
        )
    else:
        # Overlapping: sweep for the threshold minimizing total errors, and be
        # explicit that some errors are unavoidable at every threshold.
        candidates = np.unique(np.concatenate([genuine, impostor]))
        errors = [(int((genuine < t).sum() + (impostor >= t).sum()), float(t)) for t in candidates]
        best_errors, suggested = min(errors)
        typer.secho(
            f"\ndistributions overlap by {-gap:.4f} — no threshold is error-free."
            f"\n  suggested threshold: {suggested:.4f} ({best_errors} errors on "
            f"{len(genuine) + len(impostor)} pairs)",
            fg=typer.colors.YELLOW,
        )
        typer.echo("  Overlap usually means an enrollment frame was blurry or off-pose.")

    typer.echo(f"\n  use it:  tyf watch --match-threshold {suggested:.4f}\n")


# --------------------------------------------------------------------------
# ship it
# --------------------------------------------------------------------------


@app.command()
def export(
    checkpoint: Path = typer.Option(DEFAULT_RUN_DIR / "best.pt", help="Checkpoint to export."),
    out: Path | None = typer.Option(None, help="Output path. Defaults beside the checkpoint."),
    fmt: str = typer.Option("onnx", "--format", help="onnx | coreml", metavar="FMT"),
    quantize: bool = typer.Option(False, help="INT8 dynamic quantization (ONNX only)."),
    verify: bool = typer.Option(
        True,
        help="Check exported outputs match PyTorch. Disabling this is how a "
        "broken export ships silently.",
    ),
) -> None:
    """Export the liveness model for edge deployment."""
    from trainyourface.export.convert import export_model

    export_model(checkpoint, out=out, fmt=fmt, quantize=quantize, verify=verify)


@app.command()
def bench(
    model: Path | None = typer.Option(
        None, help="Model to benchmark. Defaults to the liveness model."
    ),
    runs: int = typer.Option(200, min=10, help="Timed iterations."),
    warmup: int = typer.Option(20, min=0, help="Untimed iterations first."),
    full_pipeline: bool = typer.Option(
        False, help="Benchmark detect+liveness+recognize end to end instead."
    ),
) -> None:
    """Measure latency, reporting the provider actually used.

    Reports median and p95, not mean: a mean hides the tail, and on an edge
    device the tail is what breaks a frame budget. Warmup iterations are excluded
    because the first inference includes graph optimization and kernel
    compilation, which do not recur.
    """
    from trainyourface.export.bench import run_bench

    run_bench(model=model, runs=runs, warmup=warmup, full_pipeline=full_pipeline)


@app.command()
def info() -> None:
    """Show model cache, data paths, and available execution providers."""
    import onnxruntime as ort

    from trainyourface.core.models import DETECTOR, EMBEDDER, model_dir
    from trainyourface.core.store import EnrollmentStore
    from trainyourface.liveness.predict import default_model_path

    typer.echo("\npaths")
    typer.echo(f"  models      {model_dir()}")
    typer.echo(f"  enrollments {EnrollmentStore().path}")

    typer.echo("\nweights")
    for spec in (DETECTOR, EMBEDDER):
        path = model_dir() / spec.filename
        mark = "cached" if path.exists() else "not downloaded"
        typer.echo(f"  {spec.name:<22} {mark}")
    liveness = default_model_path()
    typer.echo(
        f"  {'liveness':<22} {'found: ' + str(liveness) if liveness.exists() else 'not trained'}"
    )

    typer.echo("\nonnxruntime execution providers")
    for p in ort.get_available_providers():
        typer.echo(f"  {p}")
    typer.echo()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
