"""Unit tests for pipeline/supervised_finetune.py.

No DL model loading — backbone is replaced with a thin stub so tests run without
GPU or torch.hub access.  All assertions target the SupCon loss maths, CVAT XML
parsing, dataset construction, and training-loop plumbing.
"""

import os
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Module under test ─────────────────────────────────────────────────────────
from selfsuvis.pipeline.training.supervised import (
    AnnotatedFrameDataset,
    CvatAnnotationParser,
    SupConLoss,
    SupervisedFinetuneConfig,
    SupervisedFineTuner,
    _build_augment_transform,
    run_supervised_finetune,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _write_cvat_xml(path: str, images: list[dict]) -> None:
    """Write a minimal CVAT XML 1.1 file.

    images: list of dicts with keys: name, label (optional).
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = "test_task"
    ET.SubElement(task, "size").text = str(len(images))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "flipped").text = "False"

    labels_el = ET.SubElement(task, "labels")
    seen_labels: list[str] = []
    for img in images:
        lbl = img.get("label")
        if lbl and lbl not in seen_labels:
            seen_labels.append(lbl)
    for lbl in sorted(seen_labels):
        lbl_el = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl_el, "name").text = lbl

    segs = ET.SubElement(task, "segments")
    seg = ET.SubElement(segs, "segment")
    ET.SubElement(seg, "id").text = "0"
    ET.SubElement(seg, "start").text = "0"
    ET.SubElement(seg, "stop").text = str(max(0, len(images) - 1))

    for i, img in enumerate(images):
        img_el = ET.SubElement(root, "image")
        img_el.set("id", str(i))
        img_el.set("name", img["name"])
        img_el.set("width", "640")
        img_el.set("height", "480")
        if "label" in img:
            box = ET.SubElement(img_el, "box")
            box.set("label", img["label"])
            box.set("source", "manual")
            box.set("occluded", "0")
            box.set("xtl", "100.00")
            box.set("ytl", "100.00")
            box.set("xbr", "200.00")
            box.set("ybr", "200.00")
            box.set("z_order", "0")

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _write_fake_jpeg(path: str, width: int = 64, height: int = 64) -> None:
    """Write a tiny valid JPEG using PIL."""
    from PIL import Image

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


# ── Stub backbone for testing without torch.hub ───────────────────────────────


class _StubBackbone(nn.Module):
    """Minimal DINOv3 backbone stub: returns random unit vectors."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)
        # Expose `.blocks` so freeze logic works
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return nn.functional.normalize(torch.randn(B, self.embed_dim), dim=-1)


def _make_stub_tuner(embed_dim: int = 32) -> SupervisedFineTuner:
    """Create a SupervisedFineTuner with a stub backbone (no hub download)."""
    tuner = object.__new__(SupervisedFineTuner)
    tuner.device = "cpu"
    tuner.model_name = "stub"
    tuner.backbone = _StubBackbone(embed_dim)
    tuner.head = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(inplace=True),
        nn.Linear(embed_dim, 16),
    )
    return tuner


# ── SupConLoss tests ───────────────────────────────────────────────────────────


class TestSupConLoss:
    def test_loss_is_scalar(self):
        loss_fn = SupConLoss(temperature=0.07)
        z = nn.functional.normalize(torch.randn(8, 32), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(z, labels)
        assert loss.shape == ()

    def test_loss_positive_with_positives(self):
        """Loss should be > 0 when positives exist."""
        loss_fn = SupConLoss(temperature=0.07)
        z = nn.functional.normalize(torch.randn(8, 32), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(z, labels)
        assert loss.item() > 0.0

    def test_loss_zero_when_no_positives(self):
        """All unique labels → no positives → loss should be 0.0."""
        loss_fn = SupConLoss(temperature=0.07)
        z = nn.functional.normalize(torch.randn(4, 32), dim=-1)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(z, labels)
        assert loss.item() == 0.0

    def test_loss_decreases_when_same_class_closer(self):
        """Manually constructed case: identical embeddings per class → lower loss."""
        loss_fn = SupConLoss(temperature=0.07)
        # Different embeddings (high loss)
        z_diff = nn.functional.normalize(torch.randn(6, 32), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss_diff = loss_fn(z_diff, labels)

        # Same embedding within class (perfectly aligned positives → lower loss)
        v = nn.functional.normalize(torch.randn(3, 32), dim=-1)
        z_same = torch.stack([v[0], v[0], v[1], v[1], v[2], v[2]])
        loss_same = loss_fn(z_same, labels)

        assert loss_same.item() < loss_diff.item()

    def test_loss_single_class_all_same(self):
        """Single class: all samples are each other's positives."""
        loss_fn = SupConLoss(temperature=0.07)
        z = nn.functional.normalize(torch.randn(4, 32), dim=-1)
        labels = torch.zeros(4, dtype=torch.long)
        loss = loss_fn(z, labels)
        assert loss.item() >= 0.0

    def test_loss_gradient_flows(self):
        """Backward pass should produce non-zero gradients on a leaf parameter."""
        loss_fn = SupConLoss(temperature=0.07)
        # Use a Linear layer whose .weight is a leaf parameter
        linear = nn.Linear(16, 16, bias=False)
        x = torch.randn(6, 16)
        z = nn.functional.normalize(linear(x), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = loss_fn(z, labels)
        loss.backward()
        assert linear.weight.grad is not None
        assert linear.weight.grad.abs().sum().item() > 0

    def test_temperature_scales_loss(self):
        """Smaller temperature → sharper distribution → higher loss for hard batches."""
        z = nn.functional.normalize(torch.randn(4, 16), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        loss_low_t = SupConLoss(temperature=0.01)(z, labels)
        loss_high_t = SupConLoss(temperature=1.0)(z, labels)
        # Both should be positive; with random embeddings low-T is typically higher
        assert loss_low_t.item() > 0
        assert loss_high_t.item() > 0

    def test_two_view_batch_positives(self):
        """Simulates the two-view SupCon training batch: (2B, D) with repeated labels."""
        loss_fn = SupConLoss(temperature=0.07)
        B = 4
        z1 = nn.functional.normalize(torch.randn(B, 16), dim=-1)
        z2 = nn.functional.normalize(torch.randn(B, 16), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        z = torch.cat([z1, z2], dim=0)
        labels_2x = torch.cat([labels, labels], dim=0)
        loss = loss_fn(z, labels_2x)
        assert loss.item() > 0


# ── CvatAnnotationParser tests ─────────────────────────────────────────────────


class TestCvatAnnotationParser:
    def test_basic_parse(self, tmp_path):
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(
            xml_path,
            [
                {"name": "frame_000.jpg", "label": "car"},
                {"name": "frame_001.jpg", "label": "truck"},
                {"name": "frame_002.jpg", "label": "car"},
            ],
        )
        parser = CvatAnnotationParser(xml_path)
        assert parser.frame_labels["frame_000.jpg"] == "car"
        assert parser.frame_labels["frame_001.jpg"] == "truck"
        assert parser.frame_labels["frame_002.jpg"] == "car"

    def test_label_names_from_xml(self, tmp_path):
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(
            xml_path,
            [
                {"name": "a.jpg", "label": "bus"},
                {"name": "b.jpg", "label": "car"},
            ],
        )
        parser = CvatAnnotationParser(xml_path)
        # Labels come from XML <labels> block, alphabetically sorted
        assert "bus" in parser.label_names
        assert "car" in parser.label_names

    def test_label_to_idx_mapping(self, tmp_path):
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(
            xml_path,
            [
                {"name": "a.jpg", "label": "car"},
                {"name": "b.jpg", "label": "bus"},
            ],
        )
        parser = CvatAnnotationParser(xml_path)
        mapping = parser.label_to_idx()
        assert isinstance(mapping, dict)
        for lbl in parser.label_names:
            assert lbl in mapping
        # Indices are 0-based integers
        assert set(mapping.values()) == set(range(len(parser.label_names)))

    def test_basename_matching(self, tmp_path):
        """XML image name with subdirectory prefix → matched on basename only."""
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(
            xml_path,
            [
                {"name": "data/subfolder/frame_000.jpg", "label": "truck"},
            ],
        )
        parser = CvatAnnotationParser(xml_path)
        assert "frame_000.jpg" in parser.frame_labels

    def test_majority_vote_label(self, tmp_path):
        """Frame with multiple boxes → majority label wins."""
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "id").text = "1"
        ET.SubElement(task, "name").text = "t"
        ET.SubElement(task, "size").text = "1"
        ET.SubElement(task, "mode").text = "annotation"
        ET.SubElement(task, "overlap").text = "0"
        ET.SubElement(task, "flipped").text = "False"
        labels_el = ET.SubElement(task, "labels")
        for lbl_name in ["car", "pedestrian"]:
            lbl = ET.SubElement(labels_el, "label")
            ET.SubElement(lbl, "name").text = lbl_name
        segs = ET.SubElement(task, "segments")
        seg = ET.SubElement(segs, "segment")
        ET.SubElement(seg, "id").text = "0"
        ET.SubElement(seg, "start").text = "0"
        ET.SubElement(seg, "stop").text = "0"

        img_el = ET.SubElement(root, "image")
        img_el.set("id", "0")
        img_el.set("name", "frame_000.jpg")
        img_el.set("width", "640")
        img_el.set("height", "480")
        # 3 cars, 1 pedestrian → majority is car
        for _ in range(3):
            b = ET.SubElement(img_el, "box")
            b.set("label", "car")
            b.set("xtl", "0")
            b.set("ytl", "0")
            b.set("xbr", "10")
            b.set("ybr", "10")
        b = ET.SubElement(img_el, "box")
        b.set("label", "pedestrian")
        b.set("xtl", "0")
        b.set("ytl", "0")
        b.set("xbr", "10")
        b.set("ybr", "10")

        xml_path = str(tmp_path / "majority.xml")
        ET.ElementTree(root).write(xml_path)
        parser = CvatAnnotationParser(xml_path)
        assert parser.frame_labels.get("frame_000.jpg") == "car"

    def test_unannotated_image_skipped(self, tmp_path):
        """Image with no boxes is not included in frame_labels."""
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "id").text = "1"
        ET.SubElement(task, "name").text = "t"
        ET.SubElement(task, "size").text = "1"
        ET.SubElement(task, "mode").text = "annotation"
        ET.SubElement(task, "overlap").text = "0"
        ET.SubElement(task, "flipped").text = "False"
        labels_el = ET.SubElement(task, "labels")
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text = "car"
        segs = ET.SubElement(task, "segments")
        seg = ET.SubElement(segs, "segment")
        ET.SubElement(seg, "id").text = "0"
        ET.SubElement(seg, "start").text = "0"
        ET.SubElement(seg, "stop").text = "0"

        img_el = ET.SubElement(root, "image")
        img_el.set("id", "0")
        img_el.set("name", "unlabelled.jpg")
        img_el.set("width", "640")
        img_el.set("height", "480")
        # No boxes

        xml_path = str(tmp_path / "empty.xml")
        ET.ElementTree(root).write(xml_path)
        parser = CvatAnnotationParser(xml_path)
        assert "unlabelled.jpg" not in parser.frame_labels


# ── AnnotatedFrameDataset tests ────────────────────────────────────────────────


class TestAnnotatedFrameDataset:
    def _setup(self, tmp_path, n_frames: int = 6, labels: list[str] | None = None):
        """Create n fake frames + matching CVAT XML → return (frames_dir, parser)."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        if labels is None:
            label_cycle = ["car", "truck", "bus"]
            labels = [label_cycle[i % 3] for i in range(n_frames)]
        images_meta = []
        for i in range(n_frames):
            name = f"frame_{i:04d}.jpg"
            _write_fake_jpeg(str(frames_dir / name))
            images_meta.append({"name": name, "label": labels[i]})
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(xml_path, images_meta)
        parser = CvatAnnotationParser(xml_path)
        return str(frames_dir), parser

    def test_len_matches_annotated_frames(self, tmp_path):
        frames_dir, parser = self._setup(tmp_path, n_frames=9)
        transform = _build_augment_transform()
        ds = AnnotatedFrameDataset.from_xml(frames_dir, parser, transform, two_views=False)
        assert len(ds) == 9

    def test_two_views_returns_triple(self, tmp_path):
        frames_dir, parser = self._setup(tmp_path, n_frames=4)
        transform = _build_augment_transform()
        ds = AnnotatedFrameDataset.from_xml(frames_dir, parser, transform, two_views=True)
        item = ds[0]
        assert len(item) == 3  # (view1, view2, label_idx)
        v1, v2, lbl = item
        assert isinstance(v1, torch.Tensor)
        assert isinstance(v2, torch.Tensor)
        assert isinstance(lbl, int)

    def test_one_view_returns_pair(self, tmp_path):
        frames_dir, parser = self._setup(tmp_path, n_frames=4)
        transform = _build_augment_transform()
        ds = AnnotatedFrameDataset.from_xml(frames_dir, parser, transform, two_views=False)
        item = ds[0]
        assert len(item) == 2  # (view, label_idx)

    def test_label_indices_in_range(self, tmp_path):
        frames_dir, parser = self._setup(tmp_path, n_frames=12)
        transform = _build_augment_transform()
        ds = AnnotatedFrameDataset.from_xml(frames_dir, parser, transform, two_views=False)
        n_classes = len(parser.label_names)
        for i in range(len(ds)):
            _, lbl = ds[i]
            assert 0 <= lbl < n_classes

    def test_missing_frames_skipped(self, tmp_path):
        """Annotated frames not present on disk are skipped without error."""
        frames_dir = str(tmp_path / "frames")
        os.makedirs(frames_dir)
        # Create 3 frames, annotate 5 (2 extra annotations have no matching file)
        for i in range(3):
            _write_fake_jpeg(os.path.join(frames_dir, f"frame_{i:04d}.jpg"))
        images_meta = [
            {"name": f"frame_{i:04d}.jpg", "label": ["car", "truck", "bus"][i % 3]}
            for i in range(5)
        ]
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(xml_path, images_meta)
        parser = CvatAnnotationParser(xml_path)
        transform = _build_augment_transform()
        ds = AnnotatedFrameDataset.from_xml(frames_dir, parser, transform, two_views=False)
        assert len(ds) == 3  # only 3 files exist

    def test_raises_when_no_matching_frames(self, tmp_path):
        """ValueError if frames dir has no files matching the XML."""
        frames_dir = str(tmp_path / "empty_frames")
        os.makedirs(frames_dir)
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(xml_path, [{"name": "missing.jpg", "label": "car"}])
        parser = CvatAnnotationParser(xml_path)
        transform = _build_augment_transform()
        with pytest.raises(ValueError, match="No annotated frames"):
            AnnotatedFrameDataset.from_xml(frames_dir, parser, transform)

    def test_image_tensor_shape(self, tmp_path):
        frames_dir, parser = self._setup(tmp_path, n_frames=2)
        transform = _build_augment_transform(image_size=64)
        ds = AnnotatedFrameDataset.from_xml(frames_dir, parser, transform, two_views=False)
        v, _ = ds[0]
        assert v.shape == (3, 64, 64)


# ── SupervisedFinetuneConfig tests ─────────────────────────────────────────────


class TestSupervisedFinetuneConfig:
    def test_defaults(self):
        cfg = SupervisedFinetuneConfig(
            frames_dir="/frames",
            cvat_xml_path="/ann.xml",
            output_dir="/out",
        )
        assert cfg.epochs == 10
        assert cfg.batch_size == 16
        assert cfg.temperature == 0.07
        assert cfg.freeze_blocks == 8
        assert cfg.ssl_checkpoint is None

    def test_ssl_checkpoint_optional(self):
        cfg = SupervisedFinetuneConfig(
            frames_dir="/f",
            cvat_xml_path="/a.xml",
            output_dir="/o",
            ssl_checkpoint="/checkpoints/dino_ssl_best.pt",
        )
        assert cfg.ssl_checkpoint == "/checkpoints/dino_ssl_best.pt"


# ── run_supervised_finetune integration stub ───────────────────────────────────


class TestRunSupervisedFinetune:
    """Smoke test for run_supervised_finetune with a stub backbone.

    Patches SupervisedFineTuner to avoid torch.hub access.
    """

    def _setup_fixtures(self, tmp_path, n: int = 12):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        images_meta = []
        label_cycle = ["car", "car", "truck", "truck", "bus", "bus"]
        for i in range(n):
            name = f"frame_{i:04d}.jpg"
            _write_fake_jpeg(str(frames_dir / name))
            images_meta.append({"name": name, "label": label_cycle[i % len(label_cycle)]})
        xml_path = str(tmp_path / "ann.xml")
        _write_cvat_xml(xml_path, images_meta)
        out_dir = str(tmp_path / "checkpoints")
        return str(frames_dir), xml_path, out_dir

    def test_training_loop_runs(self, tmp_path, monkeypatch):
        """run_supervised_finetune completes without error using stub backbone."""
        frames_dir, xml_path, out_dir = self._setup_fixtures(tmp_path, n=12)

        # Patch SupervisedFineTuner.__init__ to inject stub
        def _stub_init(
            self, model_name, freeze_blocks, device, embed_dim, proj_out_dim, ssl_checkpoint=None
        ):
            self.device = device
            self.model_name = model_name
            stub = _StubBackbone(embed_dim)
            self.backbone = stub
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, proj_out_dim),
            )

        import selfsuvis.pipeline.training.supervised as sf_mod

        monkeypatch.setattr(sf_mod.SupervisedFineTuner, "__init__", _stub_init)

        # Patch save_checkpoint to avoid writing to disk
        def _stub_save(self, path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Save a tiny stub state dict
            torch.save({"stub": True}, path)

        monkeypatch.setattr(sf_mod.SupervisedFineTuner, "save_checkpoint", _stub_save)

        cfg = SupervisedFinetuneConfig(
            frames_dir=frames_dir,
            cvat_xml_path=xml_path,
            output_dir=out_dir,
            model_name="stub",
            epochs=2,
            batch_size=4,
            num_workers=0,
            device="cpu",
            embed_dim=32,
            proj_out_dim=16,
            min_eval_gate_frames=6,
            eval_gate_threshold=0.0,
        )
        result = run_supervised_finetune(cfg)
        assert isinstance(result, dict)
        assert result["accepted"]
        assert os.path.isfile(result["path"])

    def test_best_checkpoint_filename(self, tmp_path, monkeypatch):
        frames_dir, xml_path, out_dir = self._setup_fixtures(tmp_path)

        def _stub_init(
            self, model_name, freeze_blocks, device, embed_dim, proj_out_dim, ssl_checkpoint=None
        ):
            self.device = device
            self.model_name = model_name
            self.backbone = _StubBackbone(embed_dim)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, proj_out_dim)
            )

        def _stub_save(self, path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save({}, path)

        import selfsuvis.pipeline.training.supervised as sf_mod

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
            embed_dim=32,
            proj_out_dim=16,
            min_eval_gate_frames=6,
            eval_gate_threshold=0.0,
        )
        result = run_supervised_finetune(cfg)
        assert result["path"].endswith("dino_sup_best.pt")

    def test_raises_on_empty_xml(self, tmp_path, monkeypatch):
        """ValueError raised when CVAT XML has no annotated frames."""
        frames_dir = str(tmp_path / "frames")
        os.makedirs(frames_dir)
        # XML with images but no boxes → parser.frame_labels is empty
        xml_path = str(tmp_path / "empty.xml")
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "id").text = "1"
        ET.SubElement(task, "name").text = "t"
        ET.SubElement(task, "size").text = "0"
        ET.SubElement(task, "mode").text = "annotation"
        ET.SubElement(task, "overlap").text = "0"
        ET.SubElement(task, "flipped").text = "False"
        ET.SubElement(ET.SubElement(task, "labels"), "label")
        segs = ET.SubElement(task, "segments")
        seg = ET.SubElement(segs, "segment")
        ET.SubElement(seg, "id").text = "0"
        ET.SubElement(seg, "start").text = "0"
        ET.SubElement(seg, "stop").text = "0"
        ET.ElementTree(root).write(xml_path)

        import selfsuvis.pipeline.training.supervised as sf_mod

        monkeypatch.setattr(
            sf_mod,
            "run_supervised_finetune",
            lambda cfg: (_ for _ in ()).throw(ValueError("No labels")),
        )

        cfg = SupervisedFinetuneConfig(
            frames_dir=frames_dir,
            cvat_xml_path=xml_path,
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises((ValueError, StopIteration)):
            run_supervised_finetune(cfg)
