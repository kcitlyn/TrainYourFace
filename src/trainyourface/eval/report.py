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

# Below this many attack samples in a condition bucket, a rate is noise. 2 of 3
# wrong is "67% APCER", which reads as a finding and isn't one. Buckets under this
# are reported with their count and no rate.
MIN_BUCKET = 20


def stratify(
    scores: np.ndarray,
    attack_types: list[AttackType],
    conditions: list[dict],
    threshold: float,
) -> dict:
    """APCER per capture condition (illumination, environment, ...).

    This is the breakdown that turns "the model works" into "here is where it
    stops working", and it is the difference between a number and a result. A
    detector at 3% APCER overall can sit at 30% in backlit conditions — and
    backlit is precisely where someone holds a phone screen up to a camera. The
    aggregate hides exactly the case an attacker would choose.

    Same argument as worst-case-over-attack-type in `OperatingPoint.worst_apcer`,
    applied to the capture environment instead of the instrument.

    Only attack samples are bucketed. BPCER is deliberately not stratified here:
    CelebA-Spoof leaves the condition fields empty for live images (code 0 means
    "live", not a real illumination reading), so a per-condition BPCER computed
    from this would be measuring the absence of a label.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not (len(scores) == len(attack_types) == len(conditions)):
        raise ValueError(
            f"misaligned inputs: {len(scores)} scores, {len(attack_types)} types, "
            f"{len(conditions)} condition dicts"
        )

    # axis -> value -> [n_attacks, n_accepted]
    buckets: dict[str, dict[str, list[int]]] = {}
    for score, attack_type, cond in zip(scores, attack_types, conditions, strict=True):
        if not attack_type.is_attack or not cond:
            continue
        for axis, value in cond.items():
            slot = buckets.setdefault(axis, {}).setdefault(str(value), [0, 0])
            slot[0] += 1
            slot[1] += int(score < threshold)  # accepted as bona fide == missed attack

    out: dict[str, dict[str, dict]] = {}
    for axis, values in sorted(buckets.items()):
        out[axis] = {}
        for value, (n, accepted) in sorted(values.items()):
            entry: dict[str, float | int | None] = {"n": n, "missed": accepted}
            # A rate on n=3 is noise dressed as a finding, so it is withheld
            # rather than printed with a caveat nobody reads.
            entry["apcer"] = (accepted / n) if n >= MIN_BUCKET else None
            out[axis][value] = entry
    return out


def evaluate_test(
    scores: np.ndarray,
    attack_types: list[AttackType],
    val_threshold: float,
    conditions: list[dict] | None = None,
) -> dict:
    """Score the test split at a threshold chosen on validation.

    Returns both the threshold-free summary (EER/AUC, which need no operating
    point) and the operating point actually deployed.

    Args:
        conditions: optional per-sample capture conditions. When present, adds a
            per-condition APCER breakdown — see `stratify` for why the aggregate
            alone is not enough.
    """
    report = evaluate(scores, attack_types, thresholds=[val_threshold])
    deployed = apcer_bpcer(scores, attack_types, val_threshold)

    result = {
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

    if conditions and any(conditions):
        strata = stratify(scores, attack_types, conditions, val_threshold)
        if strata:
            result["by_condition"] = strata
    return result


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

    by_cond = result.get("by_condition")
    if by_cond:
        add("  APCER by capture condition")
        add("  (where the model stops working — the aggregate above hides this)")
        for axis, values in sorted(by_cond.items()):
            add(f"    {axis}")
            # Worst first: the point of this table is the weak spot, so it goes at
            # the top of each axis rather than wherever it sorts alphabetically.
            ranked = sorted(
                values.items(),
                key=lambda kv: (kv[1]["apcer"] is None, -(kv[1]["apcer"] or 0.0)),
            )
            for value, e in ranked:
                if e["apcer"] is None:
                    add(f"      {value:12s}      n/a    (n={e['n']}, too few to rate)")
                else:
                    add(f"      {value:12s} {e['apcer'] * 100:6.2f}%   (n={e['n']})")
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
    conditions = [getattr(s, "conditions", None) or {} for s in samples]
    return evaluate_test(all_scores, types, threshold, conditions=conditions)


def cross_dataset_report(source: dict, target: dict, source_name: str, target_name: str) -> str:
    """Compare in-domain and out-of-domain results side by side.

    The single most informative number a PAD project can publish, and the one
    almost none of them do. Every anti-spoofing model scores well on the dataset it
    was built from — the frames share cameras, lighting, and attack instruments, so
    the model can lean on cues specific to that capture setup. The question that
    decides whether it is deployable is what happens on a camera and a room it has
    never seen, and the gap between the two is the answer.

    Reported as a gap rather than two separate tables because presenting only the
    in-domain number is the standard way this gets oversold, and presenting only
    the out-of-domain number hides whether the model ever worked at all. A large
    drop is not a bug to hide; it is the finding. It says the model learned the
    dataset instead of the phenomenon.
    """
    lines: list[str] = []
    add = lines.append

    add("\nCross-dataset generalization")
    add("=" * 28)
    add("")
    add(f"  trained on   {source_name}")
    add(f"  in-domain    {source_name} test split")
    add(f"  out-domain   {target_name} (never seen in training)")
    add("")
    add(f"  {'metric':16s} {'in-domain':>12s} {'out-domain':>12s} {'change':>12s}")
    add(f"  {'-' * 16} {'-' * 12:>12s} {'-' * 12:>12s} {'-' * 12:>12s}")

    for key, label, scale in (
        ("eer", "EER", 100),
        ("worst_apcer", "APCER (worst)", 100),
        ("bpcer", "BPCER", 100),
        ("acer", "ACER", 100),
        ("auc", "AUC", 1),
    ):
        a, b = source.get(key), target.get(key)
        if a is None or b is None:
            continue
        unit = "%" if scale == 100 else ""
        # AUC improves upward, error rates improve downward. Signing the delta by
        # hand per metric avoids the classic mistake of calling a rising error rate
        # an improvement.
        delta = b - a
        worse = (delta > 0) if key != "auc" else (delta < 0)
        arrow = "worse" if worse and abs(delta) > 1e-9 else ("better" if abs(delta) > 1e-9 else "=")
        add(
            f"  {label:16s} {a * scale:11.2f}{unit} {b * scale:11.2f}{unit} "
            f"{delta * scale:+8.2f}{unit} {arrow}"
        )

    add("")

    # Two independent failure modes, and reading only one of them is how a broken
    # model passes review.
    #
    # DISCRIMINATION collapse shows up in EER/AUC: the model can no longer tell the
    # classes apart at any threshold.
    #
    # CALIBRATION collapse shows up in APCER/BPCER while EER stays put: the model
    # still ranks attacks above genuine faces perfectly, but the score
    # DISTRIBUTION shifted, so the threshold carried over from the source dataset
    # now sits in the wrong place. This is the more common cross-dataset failure
    # and the more dangerous one, because the threshold-free metrics look pristine.
    #
    # An early version of this function keyed the verdict on EER alone and printed
    # "the gap is small, which is the good outcome" for a model whose worst-case
    # APCER went from 0% to 100% — every single attack accepted. EER was 0.00% on
    # both sides, so by that measure nothing had changed. It was the exact class of
    # flattering-but-wrong summary this project exists to avoid.
    eer_gap = (target.get("eer") or 0) - (source.get("eer") or 0)
    apcer_gap = (target.get("worst_apcer") or 0) - (source.get("worst_apcer") or 0)
    bpcer_gap = (target.get("bpcer") or 0) - (source.get("bpcer") or 0)
    operating_gap = max(apcer_gap, bpcer_gap)

    if eer_gap > 0.10:
        add("  READ THIS: discrimination collapsed. The out-of-domain EER is more than")
        add("  10 points worse, so the model can no longer separate attacks from real")
        add("  faces at ANY threshold — it leaned on cues specific to the training")
        add("  capture setup rather than on the physics of a spoof. The in-domain")
        add("  number above is NOT a deployment estimate.")
    elif operating_gap > 0.10:
        add("  READ THIS: the threshold did not transfer. EER held up, which means the")
        add("  model still RANKS attacks above genuine faces — but the score")
        add("  distribution shifted, so the threshold carried over from the source")
        add("  dataset now sits in the wrong place and the error rates above are")
        add("  what a deployment would actually see.")
        add("")
        add("  This is a calibration failure, not a discrimination failure, and it is")
        add("  the more common one across datasets. Re-derive the threshold on data")
        add("  from the target domain (`tyf calibrate`, or a validation split of the")
        add("  target) before quoting any operating point. Do NOT re-derive it on the")
        add("  target TEST split — that is threshold-fitting on test.")
    elif eer_gap > 0.03 or operating_gap > 0.03:
        add("  The out-of-domain drop is moderate. Some of what the model learned")
        add("  transfers; quote the out-of-domain number, not the in-domain one.")
    else:
        add("  The gap is small on both discrimination and the operating point, which")
        add("  is the good outcome — but check the sample counts before reading much")
        add("  into it.")
    add("")
    return "\n".join(lines)
