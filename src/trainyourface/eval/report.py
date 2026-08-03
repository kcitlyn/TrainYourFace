"""Test-set evaluation and human-readable PAD reports.

The rule this module enforces: the decision threshold is an INPUT here, carried
over from validation. It is never recomputed on the test split.

That distinction is the difference between "our model achieves 2% ACER" and "our
model achieves 2% ACER at the best operating point we found by looking at the test
labels". The second is not a real result. To make the honest version easy and the
dishonest version awkward, `evaluate_test` requires the threshold to be passed in,
and the rendered report labels where it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trainyourface.core.contracts import AttackType
from trainyourface.eval.metrics import apcer_bpcer, evaluate


def evaluate_test(
    scores: np.ndarray,
    attack_types: list[AttackType],
    val_threshold: float,
) -> dict:
    """Score the test split at a threshold chosen on validation.

    Returns both the threshold-free summary (EER/AUC, which need no operating
    point) and the operating point actually deployed.
    """
    report = evaluate(scores, attack_types, thresholds=[val_threshold])
    deployed = apcer_bpcer(scores, attack_types, val_threshold)

    return {
        "threshold": val_threshold,
        "threshold_source": "validation split (EER point)",
        "eer": report.eer,
        "auc": report.auc,
        "deployed": deployed.model_dump(),
        "worst_apcer": deployed.worst_apcer,
        "mean_apcer": deployed.mean_apcer,
        "bpcer": deployed.bpcer,
        "acer": deployed.acer,
        "bpcer_at_apcer": report.bpcer_at_apcer,
        "n_bona_fide": report.n_bona_fide,
        "n_attack": report.n_attack,
        "attack_types_present": report.attack_types_present,
    }


def render_report(result: dict, title: str = "PAD Evaluation") -> str:
    """Format a test result as a terminal/markdown table.

    Deliberately leads with the worst-case APCER and states the sample counts. A
    report that buries n=40 behind a headline percentage invites over-reading, so
    the counts sit next to the rates.
    """
    lines: list[str] = []
    add = lines.append

    add(f"\n{title}")
    add("=" * len(title))
    add("")
    add(f"  samples          {result['n_bona_fide']} bona-fide, {result['n_attack']} attack")
    add(f"  threshold        {result['threshold']:.4f}  (from {result['threshold_source']})")
    add("")
    add("  Threshold-free")
    add(f"    EER            {result['eer'] * 100:6.2f}%")
    add(f"    AUC            {result['auc']:6.4f}")
    add("")
    add("  At the deployed threshold")
    add(f"    BPCER          {result['bpcer'] * 100:6.2f}%   (real users wrongly rejected)")
    add(f"    APCER (worst)  {result['worst_apcer'] * 100:6.2f}%   (attacks wrongly accepted)")
    add(f"    APCER (mean)   {result['mean_apcer'] * 100:6.2f}%")
    add(f"    ACER           {result['acer'] * 100:6.2f}%")
    add("")

    by_type = result["deployed"]["apcer_by_type"]
    if by_type:
        add("  APCER by attack type")
        ranked = sorted(by_type.items(), key=lambda kv: -kv[1])
        # Only flag a weakest category when one is genuinely worse than another.
        # With all types tied (including the all-zero case) there is no weak spot
        # to point at, and marking every row "weakest" is just noise.
        distinct = len({v for _, v in ranked}) > 1
        for name, val in ranked:
            marker = "  <-- weakest" if distinct and val == result["worst_apcer"] else ""
            add(f"    {name:12s}   {val * 100:6.2f}%{marker}")
        add("")

    if result.get("bpcer_at_apcer"):
        add("  Usability cost of a fixed security target")
        for target, bpcer in sorted(result["bpcer_at_apcer"].items()):
            add(f"    BPCER @ {target:12s} {bpcer * 100:6.2f}%")
        add("")

    n_total = result["n_bona_fide"] + result["n_attack"]
    if n_total < 200:
        add(f"  NOTE: n={n_total} is small. Treat these rates as indicative, not")
        add("        precise; confidence intervals are wide at this sample size.")
        add("")

    return "\n".join(lines)


def save_report(result: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))


def render_markdown_table(result: dict) -> str:
    """A compact markdown table for pasting into the README.

    Exists so the README's numbers are generated from a real eval run rather than
    typed by hand, which is how README numbers drift away from reality.
    """
    d = result
    rows = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| EER | {d['eer'] * 100:.2f}% |",
        f"| AUC | {d['auc']:.4f} |",
        f"| BPCER @ deployed threshold | {d['bpcer'] * 100:.2f}% |",
        f"| APCER (worst case) | {d['worst_apcer'] * 100:.2f}% |",
        f"| ACER | {d['acer'] * 100:.2f}% |",
    ]
    for name, val in sorted(d["deployed"]["apcer_by_type"].items()):
        rows.append(f"| APCER — {name} | {val * 100:.2f}% |")
    rows.append(f"| Test samples | {d['n_bona_fide']} bona-fide / {d['n_attack']} attack |")
    return "\n".join(rows)


def evaluate_checkpoint(
    checkpoint: Path | str,
    samples: list,
    root: Path | str,
    threshold: float,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    """Load a checkpoint and evaluate it on a sample list."""
    import torch
    from torch.utils.data import DataLoader

    from trainyourface.liveness.dataset import PADDataset
    from trainyourface.liveness.model import ModelConfig, build_model, pick_device

    dev = pick_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = PADDataset(samples, root=root, train=False, size=cfg.input_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            probs = torch.softmax(model(x.to(dev)), dim=1)[:, 1]
            scores.append(probs.cpu().numpy())

    all_scores = np.concatenate(scores) if scores else np.zeros(0)
    types = [s.attack_type for s in samples]
    if len(all_scores) != len(types):
        raise RuntimeError(f"scores ({len(all_scores)}) and samples ({len(types)}) misaligned")
    return evaluate_test(all_scores, types, threshold)
