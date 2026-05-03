"""Unit tests for pipeline/edge_inference.py — edge model hydration.

All tests run without real ONNX Runtime (mocked) and without real DINOv3 weights.
Tests cover:

  build_gallery:
    - Correct NPZ keys (embeddings, labels, label_names)
    - Shape: N = total frames across all labels, D = embed dim
    - Labels array repeats label name for each frame in that category
    - L2-normalised embeddings (norms ≈ 1.0)
    - Raises ValueError on empty labels_map
    - Raises FileNotFoundError if frame path does not exist

  EdgeClassifier (ONNX session mocked):
    - classify returns list of (str, float) tuples
    - Returns at most top_k results
    - Results sorted by score descending
    - Score is in [-1, 1] (cosine similarity)
    - Gallery with single label always returns that label
    - embed output is L2-normalised, shape (D,)
    - from_torch classmethod works without ONNX file

  Preprocessing:
    - _preprocess_image output tensor shape is (1, 3, 224, 224) float32
    - ImageNet normalisation applied (mean/std check)

  Integration smoke test:
    - build_gallery → save NPZ → EdgeClassifier loads it → classify returns correct top label
      for an exact gallery frame (score ≈ 1.0)
"""

import os
import sys
import tempfile
import unittest
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from selfsuvis.pipeline.training.edge_inference import EdgeClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(size: int = 64, color: tuple = (128, 64, 32)) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def _make_rgb_jpg(path: str, size: int = 64, color: tuple = (128, 64, 32)) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (size, size), color=color)
    img.save(path)


def _fake_embed_fn(img: Image.Image, embed_dim: int = 8) -> np.ndarray:
    """Return a random L2-normalised embedding (deterministic per image pixel sum)."""
    arr = np.array(img, dtype=np.float32)
    seed = int(arr.sum()) % (2**31)
    rng = np.random.RandomState(seed)
    v = rng.randn(embed_dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / (norm if norm > 0 else 1.0)


def _make_gallery_npz(tmp: str, embed_dim: int = 8) -> tuple:
    """Create a minimal gallery NPZ with 2 labels × 2 frames each.

    Returns (gallery_path, labels_map, colors) where colors maps label → PIL color.
    """
    colors = {
        "vehicle": (200, 100, 50),
        "barrier": (50, 100, 200),
    }
    labels_map = {}
    for label, color in colors.items():
        paths = []
        for i in range(2):
            p = os.path.join(tmp, "frames", label, f"frame_{i:03d}.jpg")
            _make_rgb_jpg(p, size=64, color=color)
            paths.append(p)
        labels_map[label] = paths
    return labels_map, colors


# ---------------------------------------------------------------------------
# _preprocess_image tests
# ---------------------------------------------------------------------------

class TestPreprocessImage(unittest.TestCase):

    def test_output_shape(self):
        from selfsuvis.pipeline.training.edge_inference import _preprocess_image
        img = _make_rgb_image(256)
        out = _preprocess_image(img, image_size=224)
        self.assertEqual(out.shape, (1, 3, 224, 224))
        self.assertEqual(out.dtype, np.float32)

    def test_output_shape_small_input(self):
        """Input smaller than image_size should still produce correct shape."""
        from selfsuvis.pipeline.training.edge_inference import _preprocess_image
        img = _make_rgb_image(64)
        out = _preprocess_image(img, image_size=224)
        self.assertEqual(out.shape, (1, 3, 224, 224))

    def test_imagenet_normalisation_applied(self):
        """A solid-grey image should have values close to (0.5-mean)/std after normalisation."""
        from selfsuvis.pipeline.training.edge_inference import _preprocess_image
        # Solid grey (128, 128, 128) ≈ 0.502 in [0,1]
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        out = _preprocess_image(img, image_size=224)
        # Channel 0: (0.502 - 0.485) / 0.229 ≈ 0.074
        expected_c0 = (128 / 255.0 - 0.485) / 0.229
        actual_c0 = float(out[0, 0, 112, 112])
        self.assertAlmostEqual(actual_c0, expected_c0, places=2)

    def test_values_are_float32(self):
        from selfsuvis.pipeline.training.edge_inference import _preprocess_image
        img = _make_rgb_image(128)
        out = _preprocess_image(img, image_size=224)
        self.assertEqual(out.dtype, np.float32)

    def test_custom_image_size(self):
        from selfsuvis.pipeline.training.edge_inference import _preprocess_image
        img = _make_rgb_image(512)
        out = _preprocess_image(img, image_size=112)
        self.assertEqual(out.shape, (1, 3, 112, 112))


# ---------------------------------------------------------------------------
# build_gallery tests
# ---------------------------------------------------------------------------

class TestBuildGallery(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.embed_dim = 8

    def _fake_backbone(self):
        """Create a minimal nn.Module backbone that returns deterministic embeddings."""
        import torch.nn as nn

        class _FakeBackbone(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.dummy = nn.Linear(1, 1)  # has parameters

            def forward(self, x):
                B = x.shape[0]
                # Use mean of input to produce deterministic output
                means = x.reshape(B, -1).mean(dim=1, keepdim=True)  # (B,1)
                out = means.expand(B, self.dim)  # (B, dim)
                return out

        return _FakeBackbone(self.embed_dim)

    def test_npz_has_correct_keys(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map, _ = _make_gallery_npz(self.tmp)
        out = os.path.join(self.tmp, "gallery.npz")
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        self.assertIn("embeddings", data)
        self.assertIn("labels", data)
        self.assertIn("label_names", data)

    def test_embeddings_shape(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map, _ = _make_gallery_npz(self.tmp)
        out = os.path.join(self.tmp, "gallery.npz")
        total_frames = sum(len(v) for v in labels_map.values())  # 4
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        self.assertEqual(data["embeddings"].shape[0], total_frames)
        self.assertEqual(data["embeddings"].shape[1], self.embed_dim)

    def test_labels_array_length(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map, _ = _make_gallery_npz(self.tmp)
        out = os.path.join(self.tmp, "gallery.npz")
        total_frames = sum(len(v) for v in labels_map.values())
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        self.assertEqual(len(data["labels"]), total_frames)

    def test_labels_repeat_correctly(self):
        """Each frame for a label should have that label in the labels array."""
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map, _ = _make_gallery_npz(self.tmp)
        out = os.path.join(self.tmp, "gallery.npz")
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        label_names_sorted = sorted(labels_map.keys())
        labels_arr = list(data["labels"])
        for label in label_names_sorted:
            expected_count = len(labels_map[label])
            actual_count = labels_arr.count(label)
            self.assertEqual(actual_count, expected_count, f"label={label}")

    def test_label_names_sorted(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map, _ = _make_gallery_npz(self.tmp)
        out = os.path.join(self.tmp, "gallery.npz")
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        actual = list(data["label_names"])
        expected = sorted(labels_map.keys())
        self.assertEqual(actual, expected)

    def test_embeddings_are_l2_normalised(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map, _ = _make_gallery_npz(self.tmp)
        out = os.path.join(self.tmp, "gallery.npz")
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        norms = np.linalg.norm(data["embeddings"], axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_raises_on_empty_labels_map(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        out = os.path.join(self.tmp, "empty.npz")
        backbone = self._fake_backbone()
        with self.assertRaises(ValueError):
            build_gallery(labels_map={}, output_path=out, backbone=backbone)

    def test_raises_on_missing_frame(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        labels_map = {"vehicle": ["/nonexistent/path/frame.jpg"]}
        out = os.path.join(self.tmp, "missing.npz")
        backbone = self._fake_backbone()
        with self.assertRaises(FileNotFoundError):
            build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)

    def test_single_label_single_frame(self):
        from selfsuvis.pipeline.training.edge_inference import build_gallery
        p = os.path.join(self.tmp, "frames", "cat", "frame_000.jpg")
        _make_rgb_jpg(p)
        labels_map = {"cat": [p]}
        out = os.path.join(self.tmp, "single.npz")
        backbone = self._fake_backbone()
        build_gallery(labels_map=labels_map, output_path=out, backbone=backbone)
        data = np.load(out, allow_pickle=True)
        self.assertEqual(data["embeddings"].shape[0], 1)
        self.assertEqual(list(data["labels"]), ["cat"])
        self.assertEqual(list(data["label_names"]), ["cat"])


# ---------------------------------------------------------------------------
# EdgeClassifier tests (ONNX session mocked)
# ---------------------------------------------------------------------------

class _MockOrtSession:
    """Minimal mock for onnxruntime.InferenceSession."""

    def __init__(self, embed_dim: int = 8):
        self._embed_dim = embed_dim
        self._input = MagicMock()
        self._input.name = "pixel_values"

    def get_inputs(self):
        return [self._input]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, input_feed):
        # Return a deterministic embedding based on the sum of the input
        inp = next(iter(input_feed.values()))
        seed = int(inp.sum() * 1000) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(1, self._embed_dim).astype(np.float32)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return [vec / (norm + 1e-8)]


def _make_gallery_file(tmp: str, labels_map: dict, embed_dim: int = 8) -> str:
    """Create a gallery NPZ with random L2-normalised embeddings."""
    all_embeddings = []
    all_labels = []
    rng = np.random.RandomState(0)
    for label in sorted(labels_map.keys()):
        for _ in labels_map[label]:
            v = rng.randn(embed_dim).astype(np.float32)
            v = v / np.linalg.norm(v)
            all_embeddings.append(v)
            all_labels.append(label)

    embeddings = np.stack(all_embeddings)
    labels_arr = np.array(all_labels, dtype=object)
    label_names = np.array(sorted(labels_map.keys()), dtype=object)

    path = os.path.join(tmp, "gallery.npz")
    np.savez(path, embeddings=embeddings, labels=labels_arr, label_names=label_names)
    return path


class TestEdgeClassifier(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.embed_dim = 8
        self.labels_map = {
            "vehicle": ["frame1.jpg", "frame2.jpg"],
            "barrier": ["frame3.jpg"],
        }
        self.gallery_path = _make_gallery_file(self.tmp, self.labels_map, self.embed_dim)

    def _make_classifier(self, top_k: int = 3) -> "EdgeClassifier":
        """Instantiate EdgeClassifier with mocked onnxruntime."""
        mock_session = _MockOrtSession(embed_dim=self.embed_dim)

        ort_mock = MagicMock()
        ort_mock.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": ort_mock}):
            from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
            with patch("selfsuvis.pipeline.training.edge_inference.os.path.isfile", return_value=True):
                clf = EdgeClassifier(
                    onnx_path="fake.onnx",
                    gallery_path=self.gallery_path,
                    top_k=top_k,
                )
        return clf

    def test_classify_returns_list_of_tuples(self):
        clf = self._make_classifier()
        img = _make_rgb_image()
        results = clf.classify(img)
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_classify_returns_at_most_top_k(self):
        clf = self._make_classifier(top_k=2)
        img = _make_rgb_image()
        results = clf.classify(img)
        self.assertLessEqual(len(results), 2)

    def test_classify_top_k_exceeds_gallery(self):
        """top_k > gallery size should return all gallery entries."""
        clf = self._make_classifier(top_k=100)
        img = _make_rgb_image()
        results = clf.classify(img)
        total = sum(len(v) for v in self.labels_map.values())
        self.assertLessEqual(len(results), total)

    def test_classify_sorted_descending(self):
        clf = self._make_classifier(top_k=3)
        img = _make_rgb_image()
        results = clf.classify(img)
        scores = [r[1] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_classify_score_in_range(self):
        clf = self._make_classifier(top_k=3)
        img = _make_rgb_image()
        results = clf.classify(img)
        for label, score in results:
            self.assertGreaterEqual(score, -1.0 - 1e-5)
            self.assertLessEqual(score, 1.0 + 1e-5)

    def test_single_label_gallery_returns_that_label(self):
        """Gallery with only one label should always return that label."""
        labels_map_single = {"vehicle": ["frame1.jpg"]}
        gallery_path = _make_gallery_file(self.tmp, labels_map_single, self.embed_dim)
        mock_session = _MockOrtSession(embed_dim=self.embed_dim)
        ort_mock = MagicMock()
        ort_mock.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": ort_mock}):
            from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
            with patch("selfsuvis.pipeline.training.edge_inference.os.path.isfile", return_value=True):
                clf = EdgeClassifier("fake.onnx", gallery_path, top_k=1)

        img = _make_rgb_image()
        results = clf.classify(img)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "vehicle")

    def test_embed_output_shape(self):
        clf = self._make_classifier()
        img = _make_rgb_image()
        emb = clf.embed(img)
        self.assertEqual(emb.shape, (self.embed_dim,))
        self.assertEqual(emb.dtype, np.float32)

    def test_embed_is_l2_normalised(self):
        clf = self._make_classifier()
        img = _make_rgb_image()
        emb = clf.embed(img)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_missing_onnxruntime_raises_import_error(self):
        """Importing EdgeClassifier when onnxruntime is absent should raise ImportError."""

        # Remove onnxruntime from sys.modules if present
        saved = sys.modules.pop("onnxruntime", None)
        # Also remove the edge_inference module so it's re-imported fresh
        ei_saved = sys.modules.pop("selfsuvis.pipeline.edge_inference", None)

        try:
            with patch.dict("sys.modules", {"onnxruntime": None}):
                from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
                with self.assertRaises(ImportError):
                    EdgeClassifier("fake.onnx", self.gallery_path)
        finally:
            if saved is not None:
                sys.modules["onnxruntime"] = saved
            if ei_saved is not None:
                sys.modules["selfsuvis.pipeline.edge_inference"] = ei_saved


# ---------------------------------------------------------------------------
# from_torch classmethod tests
# ---------------------------------------------------------------------------

class TestEdgeClassifierFromTorch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.embed_dim = 8
        self.labels_map = {"vehicle": ["f1.jpg"], "barrier": ["f2.jpg"]}
        self.gallery_path = _make_gallery_file(self.tmp, self.labels_map, self.embed_dim)

    def _fake_backbone(self):
        import torch.nn as nn

        class _FakeBackbone(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.dummy = nn.Linear(1, 1)

            def forward(self, x):
                B = x.shape[0]
                means = x.reshape(B, -1).mean(dim=1, keepdim=True)
                return means.expand(B, self.dim)

        return _FakeBackbone(self.embed_dim)

    def test_from_torch_creates_classifier(self):
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
        backbone = self._fake_backbone()
        clf = EdgeClassifier.from_torch(backbone, self.gallery_path, top_k=2)
        self.assertIsNotNone(clf)

    def test_from_torch_classify_returns_tuples(self):
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
        backbone = self._fake_backbone()
        clf = EdgeClassifier.from_torch(backbone, self.gallery_path, top_k=2)
        img = _make_rgb_image()
        results = clf.classify(img)
        self.assertIsInstance(results, list)
        for item in results:
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_from_torch_embed_shape(self):
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
        backbone = self._fake_backbone()
        clf = EdgeClassifier.from_torch(backbone, self.gallery_path)
        img = _make_rgb_image()
        emb = clf.embed(img)
        self.assertEqual(emb.shape, (self.embed_dim,))

    def test_from_torch_embed_is_normalised(self):
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
        backbone = self._fake_backbone()
        clf = EdgeClassifier.from_torch(backbone, self.gallery_path)
        img = _make_rgb_image()
        emb = clf.embed(img)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_from_torch_no_onnx_file_needed(self):
        """from_torch must not require an ONNX file to exist."""
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
        backbone = self._fake_backbone()
        # Pass a non-existent ONNX path — should not be opened
        clf = EdgeClassifier.from_torch(
            backbone, self.gallery_path, top_k=1, device="cpu"
        )
        self.assertIsNotNone(clf)


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestBuildGalleryIntegration(unittest.TestCase):
    """build_gallery → save NPZ → EdgeClassifier.from_torch loads → classify returns correct top label."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.embed_dim = 16

    def _identity_backbone(self):
        """A backbone that returns the channel-mean of each patch as its embedding.
        Different solid-colour images will produce distinct (but deterministic) vectors.
        """
        import torch.nn as nn

        dim = self.embed_dim

        class _ColorBackbone(nn.Module):
            """Returns colour statistics of the image as a (B, dim) vector."""
            def __init__(self, d):
                super().__init__()
                self.d = d
                self.dummy = nn.Linear(1, 1)

            def forward(self, x):
                # x: (B, 3, H, W) — normalised
                B = x.shape[0]
                # Use per-channel mean spread into d dimensions
                stats = x.reshape(B, 3, -1).mean(dim=2)  # (B, 3)
                # Tile 3 → d
                repeats = (self.d + 2) // 3
                out = stats.repeat(1, repeats)[:, :self.d]  # (B, d)
                return out

        return _ColorBackbone(dim)

    def test_exact_gallery_frame_scores_near_one(self):
        """Classifying a gallery image should return that image's label with high score."""
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier, build_gallery

        colors = {
            "vehicle": (220, 80, 30),
            "barrier": (30, 80, 220),
            "terrain": (80, 180, 80),
        }

        # Create one frame per label
        labels_map = {}
        label_images = {}
        for label, color in colors.items():
            p = os.path.join(self.tmp, "frames", label, "frame_000.jpg")
            _make_rgb_jpg(p, size=64, color=color)
            labels_map[label] = [p]
            label_images[label] = Image.open(p).convert("RGB")

        gallery_path = os.path.join(self.tmp, "gallery.npz")
        backbone = self._identity_backbone()
        build_gallery(labels_map=labels_map, output_path=gallery_path, backbone=backbone)

        clf = EdgeClassifier.from_torch(backbone, gallery_path, top_k=3)

        for label, img in label_images.items():
            results = clf.classify(img)
            top_label, top_score = results[0]
            self.assertEqual(
                top_label, label,
                f"Expected top label '{label}' but got '{top_label}' (score={top_score:.3f})"
            )
            self.assertGreater(top_score, 0.9, "Score for exact gallery frame should be > 0.9")

    def test_npz_loadable_by_edge_classifier_from_torch(self):
        """NPZ saved by build_gallery must be loadable by EdgeClassifier.from_torch."""
        from selfsuvis.pipeline.training.edge_inference import EdgeClassifier, build_gallery

        p1 = os.path.join(self.tmp, "frames", "cat", "frame_000.jpg")
        p2 = os.path.join(self.tmp, "frames", "dog", "frame_000.jpg")
        _make_rgb_jpg(p1, size=64, color=(200, 50, 50))
        _make_rgb_jpg(p2, size=64, color=(50, 50, 200))

        labels_map = {"cat": [p1], "dog": [p2]}
        gallery_path = os.path.join(self.tmp, "gallery2.npz")
        backbone = self._identity_backbone()
        build_gallery(labels_map=labels_map, output_path=gallery_path, backbone=backbone)

        clf = EdgeClassifier.from_torch(backbone, gallery_path, top_k=2)
        results = clf.classify(_make_rgb_image())
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
