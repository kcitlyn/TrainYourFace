"""Training loop for the PAD model.

Two decisions here are worth calling out, because both are places where the
conventional default would produce a worse model and a dishonest number.

MODEL SELECTION CRITERION
-------------------------
We select the best checkpoint by **validation EER**, not validation accuracy or
validation loss. Accuracy on an attack-heavy set rewards a model that leans toward
predicting "attack"; loss is only loosely coupled to the ranking quality we
actually care about. EER is threshold-free and directly measures how separable the
two score distributions are, which is the thing that determines whether ANY useful
operating point exists.

THRESHOLD SELECTION
-------------------
The deployment threshold is chosen on the VALIDATION split and then applied
unchanged to test. Picking the threshold on test — extremely common, and easy to do
by accident — means reporting the best-case operating point discovered using the
test labels, which is a form of test-set fitting. The gap between "EER threshold
tuned on val" and "same threshold applied to test" is itself informative, and we
print both.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from trainyourface.eval.metrics import auc_score, equal_error_rate, evaluate
from trainyourface.liveness.dataset import (
    PADDataset,
    Sample,
    class_weights,
)
from trainyourface.liveness.model import ModelConfig, build_model, count_parameters, pick_device


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    # Label smoothing: PAD labels are noisy at the boundary (a slightly blurry
    # genuine frame looks print-like), and hard targets make the model
    # overconfident on exactly those cases.
    label_smoothing: float = 0.05
    num_workers: int = 2
    seed: int = 0
    # Stop when val EER hasn't improved for this many epochs.
    patience: int = 8
    device: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_eer: float
    val_auc: float
    seconds: float


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # BatchNorm in the head needs >1 sample per batch; a trailing batch of 1
        # would raise during training.
        drop_last=shuffle,
        generator=generator if shuffle else None,
        # Deliberately NOT persistent. Workers are forked per epoch so they pick
        # up PADDataset.set_epoch(), which drives the augmentation seed. Persistent
        # workers would keep a stale copy of the dataset and every epoch would see
        # byte-identical augmented images. Costs a fork per epoch; buys actual
        # augmentation and a reproducible run.
        persistent_workers=False,
    )


def _score_split(model, loader, device, criterion=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference over a loader. Returns (spoof_scores, labels, mean_loss)."""
    import torch

    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion is not None:
                total_loss += float(criterion(logits, y))
                n_batches += 1
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.append(probs.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    return (
        np.concatenate(scores) if scores else np.zeros(0),
        np.concatenate(labels) if labels else np.zeros(0, dtype=int),
        total_loss / max(1, n_batches),
    )


def train(
    train_samples: list[Sample],
    val_samples: list[Sample],
    root: Path | str,
    out_dir: Path | str,
    config: TrainConfig | None = None,
    log=print,
    split: dict | None = None,
) -> dict:
    """Train the PAD model. Returns a summary dict; writes checkpoints to out_dir.

    Args:
        split: optional `split_fingerprint()` output, persisted into
            train_summary.json so a later `tyf eval` can prove it reconstructed
            the same held-out partition rather than assuming it did.
    """
    import torch
    import torch.nn as nn

    cfg = config or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device(cfg.device)
    model = build_model(cfg.model).to(device)
    n_params = count_parameters(model)

    log(f"device: {device}")
    log(f"parameters: {n_params:,}")
    log(f"train: {len(train_samples)} samples | val: {len(val_samples)} samples")

    train_ds = PADDataset(
        train_samples, root=root, train=True, size=cfg.model.input_size, seed=cfg.seed
    )
    val_ds = PADDataset(val_samples, root=root, train=False, size=cfg.model.input_size)

    train_loader = _loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = _loader(val_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    # Inverse-frequency class weights: without these, an attack-heavy set pushes
    # the model toward "attack", which quietly wrecks BPCER.
    w_bona, w_attack = class_weights(train_samples)
    log(f"class weights: bona_fide={w_bona:.3f} attack={w_attack:.3f}")
    weight = torch.tensor([w_bona, w_attack], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: list[EpochRecord] = []
    best_eer = float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        # Drives the augmentation seed — see PADDataset.set_epoch.
        train_ds.set_epoch(epoch)
        running, n_batches = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            # Small dataset + BatchNorm can produce occasional large gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1

        scheduler.step()
        train_loss = running / max(1, n_batches)

        val_scores, val_labels, val_loss = _score_split(model, val_loader, device, criterion)
        val_eer, _ = equal_error_rate(val_scores, val_labels.astype(bool))
        try:
            val_auc = auc_score(val_scores, val_labels.astype(bool))
        except ValueError:
            val_auc = float("nan")

        elapsed = time.perf_counter() - t0
        history.append(EpochRecord(epoch, train_loss, val_loss, val_eer, val_auc, elapsed))
        log(
            f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| val EER {val_eer:.4f} | val AUC {val_auc:.4f} | {elapsed:.1f}s"
        )

        # Select on EER, not accuracy or loss — see module docstring.
        #
        # Ties are broken by val loss, which matters more than it sounds. On a
        # small val split EER is coarsely quantized (with 42 samples it can only
        # take ~40 distinct values), so 0.0 is hit repeatedly and a naive
        # strict-improvement test locks in the FIRST such epoch — often epoch 0,
        # where the model happens to rank correctly but is barely trained and
        # poorly calibrated. Comparing loss among EER-tied epochs picks the one
        # that is actually confident, which is what the exported threshold needs.
        improved = val_eer < best_eer - 1e-9 or (
            abs(val_eer - best_eer) <= 1e-9 and val_loss < best_val_loss
        )
        if improved:
            best_eer, best_epoch, best_val_loss = val_eer, epoch, val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(cfg.model),
                    "epoch": epoch,
                    "val_eer": val_eer,
                    "n_params": n_params,
                },
                out_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                log(f"early stop: no val EER improvement in {cfg.patience} epochs")
                break

    # Choose the deployment threshold on VALIDATION, never on test.
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_scores, val_labels, _ = _score_split(model, val_loader, device)
    _, val_threshold = equal_error_rate(val_scores, val_labels.astype(bool))

    # The val loader is built with shuffle=False and drop_last=False, so scores
    # come back in `val_samples` order. That's what lets us attach per-sample
    # attack types for the per-type APCER breakdown. Asserted rather than
    # silently sliced: a length mismatch here would misalign labels to scores and
    # produce a plausible-looking but wrong per-type report.
    val_types = [s.attack_type for s in val_samples]
    if len(val_scores) != len(val_types):
        raise RuntimeError(
            f"val scores ({len(val_scores)}) and samples ({len(val_types)}) are misaligned; "
            "per-attack-type metrics would be attributed to the wrong categories"
        )
    val_report = evaluate(val_scores, val_types)

    summary = {
        "n_params": n_params,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_eer": best_eer,
        "val_threshold": val_threshold,
        "split": split,
        "val_report": val_report.model_dump(),
        "history": [asdict(h) for h in history],
        "config": {**asdict(cfg), "model": asdict(cfg.model)},
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nbest epoch {best_epoch} | val EER {best_eer:.4f} | threshold {val_threshold:.4f}")
    log(f"checkpoint: {out_dir / 'best.pt'}")
    return summary
