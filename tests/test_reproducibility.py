"""Tests that `--seed` actually means something.

These exist because of a real bug found during end-to-end verification. Training
accepted a `seed`, seeded torch and numpy, and still produced a different model on
every run: `PADDataset._augment` built its own `np.random.default_rng()` with no
seed on every single call. So `--seed 0` looked reproducible and wasn't.

The symptom was confusing rather than obvious — `tyf train` reported 0% worst-case
APCER and `tyf eval` on what looked like the same model reported 12.5%. The
numbers disagreed because a re-run had silently trained a different model.

An unreproducible training run makes every reported metric unverifiable: nobody,
including the author, can reproduce the number. So it's tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainyourface.core.contracts import AttackType
from trainyourface.liveness.dataset import PADDataset, Sample

pytestmark = pytest.mark.train  # needs torch


@pytest.fixture
def image_dir(tmp_path):
    """A few images with enough texture that augmentation visibly changes them."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):
        img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p.name)
    return tmp_path, paths


@pytest.fixture
def samples(image_dir):
    _, paths = image_dir
    return [
        Sample(
            path=p,
            attack_type=AttackType.BONA_FIDE if i % 2 else AttackType.PRINT,
            subject=f"s{i}",
        )
        for i, p in enumerate(paths)
    ]


class TestAugmentationSeeding:
    """The specific bug: augmentation ignored the seed."""

    def test_same_seed_and_epoch_gives_identical_augmentation(self, image_dir, samples):
        pytest.importorskip("torch")
        root, _ = image_dir
        a = PADDataset(samples, root=root, train=True, seed=0)
        b = PADDataset(samples, root=root, train=True, seed=0)
        a.set_epoch(3)
        b.set_epoch(3)
        assert np.array_equal(a[0][0].numpy(), b[0][0].numpy())

    def test_repeated_access_is_stable(self, image_dir, samples):
        """Reading the same item twice must not advance a hidden RNG.

        A shared generator would make item N's augmentation depend on how many
        items were read before it, so worker count would change the data.
        """
        pytest.importorskip("torch")
        root, _ = image_dir
        ds = PADDataset(samples, root=root, train=True, seed=0)
        ds.set_epoch(0)
        first = ds[0][0].numpy().copy()
        _ = ds[1][0]
        assert np.array_equal(ds[0][0].numpy(), first)

    def test_different_seed_gives_different_augmentation(self, image_dir, samples):
        pytest.importorskip("torch")
        root, _ = image_dir
        a = PADDataset(samples, root=root, train=True, seed=0)
        b = PADDataset(samples, root=root, train=True, seed=999)
        a.set_epoch(0)
        b.set_epoch(0)
        assert not np.array_equal(a[0][0].numpy(), b[0][0].numpy())

    def test_epoch_changes_augmentation(self, image_dir, samples):
        """Otherwise every epoch trains on byte-identical images and augmentation
        contributes nothing."""
        pytest.importorskip("torch")
        root, _ = image_dir
        ds = PADDataset(samples, root=root, train=True, seed=0)
        ds.set_epoch(0)
        e0 = ds[0][0].numpy().copy()
        ds.set_epoch(1)
        assert not np.array_equal(e0, ds[0][0].numpy())

    def test_eval_mode_never_augments(self, image_dir, samples):
        """Validation/test images must be untouched, or the metric measures noise."""
        pytest.importorskip("torch")
        root, _ = image_dir
        ds = PADDataset(samples, root=root, train=False, seed=0)
        ds.set_epoch(0)
        first = ds[0][0].numpy().copy()
        ds.set_epoch(7)
        assert np.array_equal(first, ds[0][0].numpy())


class TestPreprocessingIsShared:
    """Training and inference must normalize identically.

    The original project had exactly this bug in its recognition path: enrollment
    used an aligned crop, the live path an unaligned one, so the two embedding
    sets weren't comparable. Nothing errored — matching just quietly degraded.
    """

    def test_dataset_and_inference_agree(self, image_dir, samples):
        pytest.importorskip("torch")
        from trainyourface.liveness.dataset import load_image, to_model_input

        root, paths = image_dir
        ds = PADDataset(samples, root=root, train=False)
        from_dataset = ds[0][0].numpy()
        from_inference = to_model_input(load_image(root / paths[0]))
        assert np.array_equal(from_dataset, from_inference)

    def test_normalization_range(self):
        from trainyourface.liveness.dataset import to_model_input

        white = to_model_input(np.full((8, 8, 3), 255, dtype=np.uint8))
        black = to_model_input(np.zeros((8, 8, 3), dtype=np.uint8))
        assert white.max() == pytest.approx(1.0, abs=0.01)
        assert black.min() == pytest.approx(-1.0, abs=0.01)

    def test_bgr_to_rgb_channel_order(self):
        """A silent BGR/RGB swap costs accuracy without ever raising."""
        from trainyourface.liveness.dataset import to_model_input

        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # blue in BGR
        out = to_model_input(bgr)
        # Channel 2 of CHW output should be the blue one after conversion.
        assert out[2].mean() > out[0].mean()

    def test_output_is_channels_first(self):
        from trainyourface.liveness.dataset import to_model_input

        assert to_model_input(np.zeros((16, 16, 3), dtype=np.uint8)).shape == (3, 16, 16)
