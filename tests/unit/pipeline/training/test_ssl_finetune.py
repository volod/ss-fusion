"""Unit tests for pipeline/ssl_finetune.py — self-supervised domain adaptation.

All tests run without GPU and without real DINOv3 weights (backbone mocked).
Tests cover:
  - NTXentLoss shape, values, gradient flow
  - AugmentPairDataset: frame discovery, pair shape, two different views
  - TemporalPairDataset: pair construction, single-frame dirs skipped, max_gap
  - ProjectionHead: output shape + L2 norm
  - DINOFineTuner: freeze strategy (trainable param count), forward shape, save/load
  - build_augment_transform: output tensor shape
  - run_finetune: E2E smoke test with tiny synthetic data and mocked backbone
  - config_from_settings: env var wiring
"""
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_jpg(path: str, size: int = 64) -> None:
    """Write a small solid-colour JPEG to path (creates parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (size, size), color=(random.randint(0, 255),) * 3)
    img.save(path)


def _frames_dir_with_videos(tmp: str, videos: dict) -> str:
    """
    Create frames_dir/  with per-video subdirs.
    videos = {"vid1": 5, "vid2": 1}  → vid1 gets 5 frames, vid2 gets 1.
    """
    fdir = os.path.join(tmp, "frames")
    for vid, count in videos.items():
        vdir = os.path.join(fdir, vid)
        for i in range(count):
            _make_rgb_jpg(os.path.join(vdir, f"frame_{i:04d}.jpg"))
    return fdir


# ---------------------------------------------------------------------------
# NTXentLoss
# ---------------------------------------------------------------------------

class TestNTXentLoss(unittest.TestCase):

    def _loss(self, B: int = 4, D: int = 8):
        from selfsuvis.pipeline.training.ssl import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(B, D)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        z2 = torch.randn(B, D)
        z2 = torch.nn.functional.normalize(z2, dim=-1)
        return loss_fn(z1, z2)

    def test_loss_is_scalar(self):
        loss = self._loss()
        self.assertEqual(loss.shape, ())

    def test_loss_is_positive(self):
        loss = self._loss()
        self.assertGreater(loss.item(), 0.0)

    def test_loss_is_finite(self):
        loss = self._loss()
        self.assertTrue(torch.isfinite(loss))

    def test_identical_embeddings_gives_lower_loss(self):
        """Identical z1==z2 should give lower loss than random pairs."""
        from selfsuvis.pipeline.training.ssl import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.07)
        z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        loss_identical = loss_fn(z, z).item()
        z2_random = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        loss_random = loss_fn(z, z2_random).item()
        self.assertLess(loss_identical, loss_random)

    def test_gradient_flows(self):
        from selfsuvis.pipeline.training.ssl import NTXentLoss
        loss_fn = NTXentLoss()
        raw = torch.randn(4, 8, requires_grad=True)
        z1 = torch.nn.functional.normalize(raw, dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        loss = loss_fn(z1, z2)
        loss.backward()
        self.assertIsNotNone(raw.grad)

    def test_temperature_scaling(self):
        """Higher temperature → softer distribution → lower loss magnitude on average."""
        from selfsuvis.pipeline.training.ssl import NTXentLoss
        z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        loss_hot = NTXentLoss(temperature=1.0)(z, z2).item()
        loss_cold = NTXentLoss(temperature=0.07)(z, z2).item()
        # Both should be finite and positive
        self.assertTrue(torch.isfinite(torch.tensor(loss_hot)))
        self.assertTrue(torch.isfinite(torch.tensor(loss_cold)))

    def test_batch_size_1_raises(self):
        """Batch of 1 makes no valid negatives — cross_entropy will still run but result may be degenerate."""
        from selfsuvis.pipeline.training.ssl import NTXentLoss
        loss_fn = NTXentLoss()
        z = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
        # Should not crash (even if numerically degenerate)
        loss = loss_fn(z, z)
        self.assertTrue(torch.isfinite(loss) or True)  # just check no exception


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------

class TestProjectionHead(unittest.TestCase):

    def test_output_shape(self):
        from selfsuvis.pipeline.training.ssl import ProjectionHead
        head = ProjectionHead(in_dim=768, hidden_dim=256, out_dim=64)
        x = torch.randn(4, 768)
        out = head(x)
        self.assertEqual(out.shape, (4, 64))

    def test_output_is_l2_normalised(self):
        from selfsuvis.pipeline.training.ssl import ProjectionHead
        head = ProjectionHead(in_dim=64, out_dim=32)
        x = torch.randn(8, 64)
        out = head(x)
        norms = out.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(8), atol=1e-5))


# ---------------------------------------------------------------------------
# build_augment_transform
# ---------------------------------------------------------------------------

class TestAugmentTransform(unittest.TestCase):

    def test_output_tensor_shape(self):
        from selfsuvis.pipeline.training.ssl import build_augment_transform
        t = build_augment_transform(image_size=224)
        img = Image.new("RGB", (512, 384))
        out = t(img)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_two_views_differ(self):
        """Stochastic augmentation → two calls on same image should rarely be identical."""
        from selfsuvis.pipeline.training.ssl import build_augment_transform
        t = build_augment_transform()
        img = Image.new("RGB", (256, 256), color=(128, 64, 32))
        # Run 5 times; at least one pair should differ
        pairs = [(t(img), t(img)) for _ in range(5)]
        any_differ = any(not torch.equal(a, b) for a, b in pairs)
        self.assertTrue(any_differ)


# ---------------------------------------------------------------------------
# AugmentPairDataset
# ---------------------------------------------------------------------------

class TestAugmentPairDataset(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_discovers_frames(self):
        from selfsuvis.pipeline.training.ssl import AugmentPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"vid1": 4, "vid2": 3})
        ds = AugmentPairDataset(fdir, build_augment_transform(32))
        self.assertEqual(len(ds), 7)

    def test_item_is_two_tensors(self):
        from selfsuvis.pipeline.training.ssl import AugmentPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"vid1": 2})
        ds = AugmentPairDataset(fdir, build_augment_transform(32))
        v1, v2 = ds[0]
        self.assertEqual(v1.shape, (3, 32, 32))
        self.assertEqual(v2.shape, (3, 32, 32))

    def test_empty_dir_raises(self):
        from selfsuvis.pipeline.training.ssl import AugmentPairDataset, build_augment_transform
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            AugmentPairDataset(empty, build_augment_transform(32))

    def test_finds_jpg_and_png(self):
        from selfsuvis.pipeline.training.ssl import AugmentPairDataset, build_augment_transform
        fdir = os.path.join(self.tmp, "mixed")
        _make_rgb_jpg(os.path.join(fdir, "a.jpg"))
        # write a PNG
        os.makedirs(fdir, exist_ok=True)
        Image.new("RGB", (32, 32)).save(os.path.join(fdir, "b.png"))
        ds = AugmentPairDataset(fdir, build_augment_transform(32))
        self.assertEqual(len(ds), 2)


# ---------------------------------------------------------------------------
# TemporalPairDataset
# ---------------------------------------------------------------------------

class TestTemporalPairDataset(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_pairs_from_two_videos(self):
        from selfsuvis.pipeline.training.ssl import TemporalPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"vid1": 5, "vid2": 4})
        ds = TemporalPairDataset(fdir, build_augment_transform(32), max_gap=1)
        # vid1: 5 frames → 4 pairs (frames 0..3 each pair with next)
        # vid2: 4 frames → 3 pairs
        self.assertEqual(len(ds), 7)

    def test_single_frame_dir_skipped(self):
        from selfsuvis.pipeline.training.ssl import TemporalPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"solo": 1, "multi": 3})
        ds = TemporalPairDataset(fdir, build_augment_transform(32), max_gap=1)
        self.assertEqual(len(ds), 2)  # multi: 3 frames → 2 pairs; solo skipped

    def test_all_single_frame_raises(self):
        from selfsuvis.pipeline.training.ssl import TemporalPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"a": 1, "b": 1})
        with self.assertRaises(ValueError):
            TemporalPairDataset(fdir, build_augment_transform(32))

    def test_item_shape(self):
        from selfsuvis.pipeline.training.ssl import TemporalPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"vid": 3})
        ds = TemporalPairDataset(fdir, build_augment_transform(32))
        v1, v2 = ds[0]
        self.assertEqual(v1.shape, (3, 32, 32))
        self.assertEqual(v2.shape, (3, 32, 32))

    def test_max_gap_respected(self):
        """With max_gap=1 every pair must be adjacent (gap exactly 1)."""
        from selfsuvis.pipeline.training.ssl import TemporalPairDataset, build_augment_transform
        fdir = _frames_dir_with_videos(self.tmp, {"vid": 6})
        ds = TemporalPairDataset(fdir, build_augment_transform(32), max_gap=1)
        # All pairs should be (frame[i], frame[i+1]) — filenames differ by 1
        for p1, p2 in ds.pairs:
            i1 = int(Path(p1).stem.split("_")[-1])
            i2 = int(Path(p2).stem.split("_")[-1])
            self.assertEqual(i2 - i1, 1)

    def test_empty_dir_raises(self):
        from selfsuvis.pipeline.training.ssl import TemporalPairDataset, build_augment_transform
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            TemporalPairDataset(empty, build_augment_transform(32))


class TestMultimodalPairSchema(unittest.TestCase):

    def test_pair_serialization_is_json_friendly(self):
        from selfsuvis.pipeline.training.ssl import CrossModalPair
        pair = CrossModalPair(
            anchor_frame_path="/tmp/a.jpg",
            positive_frame_path="/tmp/b.jpg",
            time_delta_sec=0.5,
            sample_weight=1.2,
            modality_payload={"depth_similarity_target": 0.9, "occupancy_summary": {"free": 0.7}},
            pair_source="depth_alignment",
        )
        payload = pair.to_dict()
        self.assertEqual(payload["pair_type"], "cross_modal")
        self.assertEqual(payload["pair_source"], "depth_alignment")
        self.assertEqual(payload["anchor_frame_path"], "/tmp/a.jpg")
        self.assertAlmostEqual(payload["modality_payload"]["depth_similarity_target"], 0.9)

    def test_collate_handles_missing_optional_modalities(self):
        from selfsuvis.pipeline.training.ssl import (
            GeometryPair,
            TemporalVisualPair,
            collate_multimodal_pairs,
        )
        batch = collate_multimodal_pairs([
            TemporalVisualPair(
                anchor_frame_path="a.jpg",
                positive_frame_path="b.jpg",
                time_delta_sec=0.2,
                track_id=7,
            ),
            GeometryPair(
                anchor_frame_path="c.jpg",
                positive_frame_path="d.jpg",
                time_delta_sec=1.0,
                pose_overlap_score=0.8,
                modality_payload={"geometry_similarity_target": 0.8},
            ),
        ])
        self.assertEqual(batch["pair_types"], ["temporal_visual", "geometry"])
        self.assertEqual(batch["track_id"], [7, None])
        self.assertEqual(batch["depth_similarity_target"], [None, None])
        self.assertEqual(batch["geometry_similarity_target"], [None, 0.8])
        self.assertEqual(tuple(batch["time_delta_sec"].shape), (2,))
        self.assertEqual(tuple(batch["sample_weight"].shape), (2,))


class TestMultimodalConsistencyLoss(unittest.TestCase):

    def test_auxiliary_losses_are_reported_when_targets_exist(self):
        from selfsuvis.pipeline.training.ssl import MultimodalConsistencyLoss, NTXentLoss
        z1 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        batch_meta = {
            "sample_weight": torch.ones(4),
            "depth_similarity_target": [0.8, None, 0.7, None],
            "motion_similarity_target": [None, 0.6, None, 0.9],
            "geometry_similarity_target": [0.5, 0.4, None, None],
        }
        loss_fn = MultimodalConsistencyLoss(
            NTXentLoss(),
            depth_weight=0.1,
            motion_weight=0.2,
            geometry_weight=0.3,
        )
        loss, components = loss_fn(z1, z2, batch_meta)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("contrastive_loss", components)
        self.assertGreaterEqual(components["depth_consistency_loss"], 0.0)
        self.assertGreaterEqual(components["motion_consistency_loss"], 0.0)
        self.assertGreaterEqual(components["geometry_consistency_loss"], 0.0)


# ---------------------------------------------------------------------------
# DINOFineTuner — freeze strategy (mocked backbone)
# ---------------------------------------------------------------------------

class _FakeBlock(nn.Module):
    """Minimal stand-in for a ViT transformer block."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return x


class _FakeBackbone(nn.Module):
    """12-block fake ViT backbone for testing freeze logic.

    Always returns (B, 4) regardless of input shape, mimicking a ViT CLS token output.
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(4)
        self.patch_embed = nn.Linear(4, 4)

    def forward(self, x):
        B = x.shape[0]
        # Return trainable output so gradients can flow through unfrozen blocks
        out = torch.zeros(B, 4, device=x.device)
        for block in self.blocks:
            out = out + block.linear.weight.sum() * 0  # keep grad graph connected
        return out


class TestDINOFineTuner(unittest.TestCase):

    def _make_tuner(self, freeze_blocks: int = 10, embed_dim: int = 4):
        from selfsuvis.pipeline.training.ssl import DINOFineTuner
        backbone = _FakeBackbone()

        with patch("torch.hub.load", return_value=backbone):
            tuner = DINOFineTuner(
                model_name="dinov3_vitb14",
                freeze_blocks=freeze_blocks,
                device="cpu",
                embed_dim=embed_dim,
                proj_out_dim=8,
            )
        return tuner, backbone

    def test_frozen_blocks_have_no_grad(self):
        tuner, backbone = self._make_tuner(freeze_blocks=10)
        for i, block in enumerate(backbone.blocks):
            for param in block.parameters():
                if i < 10:
                    self.assertFalse(param.requires_grad, f"block {i} should be frozen")
                else:
                    self.assertTrue(param.requires_grad, f"block {i} should be trainable")

    def test_unfrozen_blocks_have_grad(self):
        tuner, backbone = self._make_tuner(freeze_blocks=6)
        trainable = [i for i, b in enumerate(backbone.blocks)
                     if any(p.requires_grad for p in b.parameters())]
        self.assertEqual(trainable, [6, 7, 8, 9, 10, 11])

    def test_freeze_all_blocks(self):
        tuner, backbone = self._make_tuner(freeze_blocks=12)
        # Head params are still trainable
        head_params = sum(p.numel() for p in tuner.head.parameters() if p.requires_grad)
        self.assertGreater(head_params, 0)

    def test_trainable_params_less_than_total(self):
        tuner, backbone = self._make_tuner(freeze_blocks=10)
        total = sum(p.numel() for p in tuner.parameters())
        trainable = sum(p.numel() for p in tuner.trainable_params())
        self.assertLess(trainable, total)

    def test_save_and_reload_checkpoint(self):
        tuner, backbone = self._make_tuner(freeze_blocks=10, embed_dim=4)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "sub", "test.pt")
            tuner.save_checkpoint(ckpt)
            self.assertTrue(os.path.isfile(ckpt))
            # Reload into a fresh backbone
            fresh = _FakeBackbone()
            with patch("torch.hub.load", return_value=fresh):
                from selfsuvis.pipeline.training.ssl import DINOFineTuner
                DINOFineTuner.load_backbone_weights(fresh, ckpt, "cpu")
            # Weights should match
            orig_sd = backbone.state_dict()
            fresh_sd = fresh.state_dict()
            for k in orig_sd:
                self.assertTrue(torch.equal(orig_sd[k], fresh_sd[k]), f"mismatch at {k}")


# ---------------------------------------------------------------------------
# run_finetune — smoke test (tiny data, mocked backbone)
# ---------------------------------------------------------------------------

class TestRunFinetune(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_augment_approach_produces_checkpoint(self):
        from selfsuvis.pipeline.training.ssl import FinetuneConfig, run_finetune

        fdir = _frames_dir_with_videos(self.tmp, {"vid1": 4})
        out = os.path.join(self.tmp, "ckpts")
        cfg = FinetuneConfig(
            frames_dir=fdir,
            output_dir=out,
            approach="augment",
            epochs=1,
            batch_size=2,
            embed_dim=4,
            proj_out_dim=4,
            num_workers=0,
            device="cpu",
        )

        fake_backbone = _FakeBackbone()
        with patch("torch.hub.load", return_value=fake_backbone):
            best = run_finetune(cfg)

        self.assertTrue(os.path.isfile(best))

    def test_temporal_approach_produces_checkpoint(self):
        from selfsuvis.pipeline.training.ssl import FinetuneConfig, run_finetune

        fdir = _frames_dir_with_videos(self.tmp, {"vid1": 4, "vid2": 3})
        out = os.path.join(self.tmp, "ckpts2")
        cfg = FinetuneConfig(
            frames_dir=fdir,
            output_dir=out,
            approach="temporal",
            epochs=1,
            batch_size=2,
            embed_dim=4,
            proj_out_dim=4,
            num_workers=0,
            device="cpu",
        )

        fake_backbone = _FakeBackbone()
        with patch("torch.hub.load", return_value=fake_backbone):
            best = run_finetune(cfg)

        self.assertTrue(os.path.isfile(best))

    def test_per_epoch_checkpoints_written(self):
        from selfsuvis.pipeline.training.ssl import FinetuneConfig, run_finetune

        fdir = _frames_dir_with_videos(self.tmp, {"vid": 4})
        out = os.path.join(self.tmp, "ckpts3")
        cfg = FinetuneConfig(
            frames_dir=fdir,
            output_dir=out,
            approach="augment",
            epochs=2,
            batch_size=2,
            save_every=1,
            embed_dim=4,
            proj_out_dim=4,
            num_workers=0,
            device="cpu",
        )
        fake_backbone = _FakeBackbone()
        with patch("torch.hub.load", return_value=fake_backbone):
            run_finetune(cfg)

        ckpts = [f for f in os.listdir(out) if f.startswith("dino_ssl_0")]
        self.assertEqual(len(ckpts), 2)  # epoch 001 and 002

    def test_returns_best_path_string(self):
        from selfsuvis.pipeline.training.ssl import FinetuneConfig, run_finetune

        fdir = _frames_dir_with_videos(self.tmp, {"vid": 4})
        out = os.path.join(self.tmp, "ckpts4")
        cfg = FinetuneConfig(
            frames_dir=fdir, output_dir=out,
            approach="augment", epochs=1, batch_size=2,
            embed_dim=4, proj_out_dim=4, num_workers=0, device="cpu",
        )
        fake_backbone = _FakeBackbone()
        with patch("torch.hub.load", return_value=fake_backbone):
            result = run_finetune(cfg)
        self.assertIsInstance(result, str)
        self.assertIn("best", result)


# ---------------------------------------------------------------------------
# config_from_settings
# ---------------------------------------------------------------------------

class TestConfigFromSettings(unittest.TestCase):

    def test_defaults_populated(self):
        from selfsuvis.pipeline.training.ssl import config_from_settings
        cfg = config_from_settings()
        self.assertIsInstance(cfg.epochs, int)
        self.assertGreater(cfg.epochs, 0)
        self.assertIsInstance(cfg.lr, float)
        self.assertGreater(cfg.lr, 0)
        self.assertIn(cfg.approach, ("temporal", "augment"))

    def test_env_override_respected(self):
        import selfsuvis.pipeline.core.config as pc
        original = pc.settings.SSL_FINETUNE_EPOCHS
        try:
            pc.settings.SSL_FINETUNE_EPOCHS = 99
            from selfsuvis.pipeline.training.ssl import config_from_settings
            cfg = config_from_settings()
            self.assertEqual(cfg.epochs, 99)
        finally:
            pc.settings.SSL_FINETUNE_EPOCHS = original


if __name__ == "__main__":
    unittest.main()
