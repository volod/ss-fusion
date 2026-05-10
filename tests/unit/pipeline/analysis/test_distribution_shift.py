"""Unit tests for pipeline.supervised_finetune._eval_distribution_shift.

No GPU or torch.hub access — backbone is a tiny stub Linear layer.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from selfsuvis.pipeline.training.supervised import _eval_distribution_shift

# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_fake_jpeg(path: str, color: tuple = (128, 64, 32)) -> None:
    arr = np.full((32, 32, 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


class _StubBackbone(nn.Module):
    """Minimal backbone stub with .blocks for freeze logic compatibility."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = torch.randn(B, self.embed_dim)
        return nn.functional.normalize(out, dim=-1)


class _RandomBackbone(nn.Module):
    """Returns normalised random vectors (no consistent class structure)."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = torch.randn(B, self.embed_dim)
        return nn.functional.normalize(out, dim=-1)


class _ClusteredBackbone(nn.Module):
    """Returns a fixed class vector based on call order.

    Items are assumed to arrive in class-contiguous order (all class-0 first,
    then all class-1, etc.) with ``n_per_class`` items per class — matching the
    order produced by ``_make_eval_items``.
    """

    def __init__(self, class_vecs: torch.Tensor, n_per_class: int):
        super().__init__()
        self.register_buffer("class_vecs", class_vecs)
        self.n_per_class = n_per_class
        self._call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_classes = self.class_vecs.shape[0]
        idx = min(self._call_count // self.n_per_class, n_classes - 1)
        self._call_count += 1
        return self.class_vecs[idx : idx + 1]  # (1, D)


def _make_eval_items(tmp_dir: str, n_per_class: int = 4, n_classes: int = 2):
    """Produce items in class-contiguous order: [c0]*n, [c1]*n, ..."""
    items = []
    for c in range(n_classes):
        for i in range(n_per_class):
            p = os.path.join(tmp_dir, f"c{c}_f{i}.jpg")
            _write_fake_jpeg(p)
            items.append((p, c))
    return items


_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestEvalDistributionShift:
    def test_returns_float(self, tmp_path):
        items = _make_eval_items(str(tmp_path))
        torch.manual_seed(0)
        backbone = _RandomBackbone(16)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        assert isinstance(result, float)

    def test_returns_zero_for_too_few_samples(self, tmp_path):
        """< 4 samples total → returns 0.0 without error."""
        items = _make_eval_items(str(tmp_path), n_per_class=1)  # 2 items total
        backbone = _RandomBackbone(16)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        assert result == 0.0

    def test_returns_zero_for_single_class(self, tmp_path):
        """Single class → no inter-class pairs → returns 0.0."""
        items = _make_eval_items(str(tmp_path), n_per_class=4, n_classes=1)
        backbone = _RandomBackbone(16)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        assert result == 0.0

    def test_result_in_valid_range(self, tmp_path):
        """Gap = mean(intra) − mean(inter); both means are cosine sims ∈ [−1, 1]."""
        items = _make_eval_items(str(tmp_path), n_per_class=4)
        torch.manual_seed(0)
        backbone = _RandomBackbone(16)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        assert -2.0 <= result <= 2.0

    def test_high_gap_for_perfectly_clustered_backbone(self, tmp_path):
        """Perfectly clustered backbone → intra-class cosine = 1, inter = 0 → gap ≈ 1."""
        embed_dim = 16
        n_per_class = 4
        # Two orthogonal unit vectors — perfect class separation
        v0 = torch.zeros(embed_dim)
        v0[0] = 1.0
        v1 = torch.zeros(embed_dim)
        v1[1] = 1.0
        class_vecs = torch.stack([v0, v1])

        items = _make_eval_items(str(tmp_path), n_per_class=n_per_class)
        backbone = _ClusteredBackbone(class_vecs, n_per_class=n_per_class)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        # Orthogonal → intra=1.0, inter=0.0, gap=1.0
        assert result > 0.8, f"Expected gap > 0.8 for perfectly clustered backbone, got {result}"

    def test_low_gap_for_random_backbone(self, tmp_path):
        """Random backbone → no class structure → gap near 0."""
        torch.manual_seed(42)
        items = _make_eval_items(str(tmp_path), n_per_class=8)
        backbone = _RandomBackbone(64)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        # Random unit vectors: expected intra ≈ inter ≈ 0, gap ≈ 0
        assert abs(result) < 0.6, f"Expected near-zero gap for random backbone, got {result}"

    def test_skips_unreadable_images(self, tmp_path):
        """Missing image files are silently skipped."""
        items = _make_eval_items(str(tmp_path), n_per_class=4)
        items.append(("/nonexistent/missing.jpg", 0))
        backbone = _RandomBackbone(16)
        result = _eval_distribution_shift(backbone, items, "cpu", _TRANSFORM)
        assert isinstance(result, float)

    def test_distribution_shift_key_in_run_result(self, tmp_path, monkeypatch):
        """run_supervised_finetune result dict always contains 'distribution_shift' key."""
        import os
        from xml.etree import ElementTree as ET

        import selfsuvis.pipeline.training.supervised as sf_mod
        from selfsuvis.pipeline.training.supervised import (
            SupervisedFinetuneConfig,
            run_supervised_finetune,
        )

        frames_dir = str(tmp_path / "frames")
        os.makedirs(frames_dir)
        out_dir = str(tmp_path / "out")

        # Write 8 labelled frames (4 per class) + minimal CVAT XML
        images = []
        for label in ("car", "truck"):
            for i in range(4):
                name = f"{label}_{i}.jpg"
                _write_fake_jpeg(os.path.join(frames_dir, name))
                images.append({"name": name, "label": label})

        xml_path = str(tmp_path / "ann.xml")
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        for k, v in [
            ("id", "1"),
            ("name", "t"),
            ("size", str(len(images))),
            ("mode", "annotation"),
            ("overlap", "0"),
            ("flipped", "False"),
        ]:
            ET.SubElement(task, k).text = v
        labels_el = ET.SubElement(task, "labels")
        for lbl in ("car", "truck"):
            ET.SubElement(ET.SubElement(labels_el, "label"), "name").text = lbl
        segs = ET.SubElement(task, "segments")
        seg = ET.SubElement(segs, "segment")
        for k, v in [("id", "0"), ("start", "0"), ("stop", str(len(images) - 1))]:
            ET.SubElement(seg, k).text = v
        for i, img in enumerate(images):
            el = ET.SubElement(root, "image")
            el.set("id", str(i))
            el.set("name", img["name"])
            el.set("width", "32")
            el.set("height", "32")
            box = ET.SubElement(el, "box")
            for k, v in [
                ("label", img["label"]),
                ("source", "manual"),
                ("occluded", "0"),
                ("xtl", "0"),
                ("ytl", "0"),
                ("xbr", "16"),
                ("ybr", "16"),
                ("z_order", "0"),
            ]:
                box.set(k, v)
        ET.ElementTree(root).write(xml_path)

        embed_dim = 16
        proj_out_dim = 8

        def _stub_init(
            self, model_name, freeze_blocks, device, embed_dim, proj_out_dim, ssl_checkpoint=None
        ):
            self.device = device
            self.model_name = model_name
            self.backbone = _StubBackbone(embed_dim)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, proj_out_dim),
            )

        def _stub_save(self, path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({}, path)

        monkeypatch.setattr(sf_mod.SupervisedFineTuner, "__init__", _stub_init)
        monkeypatch.setattr(sf_mod.SupervisedFineTuner, "save_checkpoint", _stub_save)

        cfg = SupervisedFinetuneConfig(
            frames_dir=frames_dir,
            cvat_xml_path=xml_path,
            output_dir=out_dir,
            model_name="stub",
            epochs=1,
            batch_size=4,
            num_workers=0,
            device="cpu",
            embed_dim=embed_dim,
            proj_out_dim=proj_out_dim,
            min_eval_gate_frames=4,
            eval_gate_threshold=0.0,
        )
        result = run_supervised_finetune(cfg)
        assert "distribution_shift" in result, (
            f"Expected 'distribution_shift' key in result, got keys: {list(result.keys())}"
        )
        assert isinstance(result["distribution_shift"], float)
        assert -2.0 <= result["distribution_shift"] <= 2.0
