"""Tests for Stage 2 distillation: ViT-S/14 → EfficientViT-B1.

Covers:
  - DistillConfig.stage field defaults and stage=2 setting
  - DistillConfig.lambda_caption_anchor field exists (TODOS confirmation)
  - KnowledgeDistiller._load_student() EfficientViT branch
  - KnowledgeDistiller._load_student() ImportError when timm/EfficientViT unavailable
  - run_distillation_efficientvit() enforces lambda_rkd_a=0.0 regardless of caller config
  - export_efficientvit_onnx() calls torch.onnx.export with correct args
  - export_efficientvit_onnx() returns the output path and creates parent dirs
  - step_distill_stage2() skips when no Stage 1 backbone
  - step_distill_stage2() skips when run_distillation_efficientvit raises
  - step_distill_stage2() skips when best_path is missing after distillation
  - step_distill_stage2() succeeds and returns student_backbone + ckpt_mb
  - step_distill_stage2() handles ONNX export failure gracefully (not skipped)
  - step_distill_stage2() passes stage=2 and lambda_rkd_a=0 in config
"""

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# ── Shared tiny model that accepts any spatial input ─────────────────────────


class _AnyModel(nn.Module):
    """Returns a constant tensor of shape (B, out_dim) regardless of input size."""

    def __init__(self, out_dim: int = 384) -> None:
        super().__init__()
        self._w = nn.Parameter(torch.ones(out_dim))
        self._out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._w.unsqueeze(0).expand(x.shape[0], self._out_dim)


def _fake_distiller_stats(best_path: str) -> dict:
    mock_distiller = MagicMock()
    mock_distiller.student_backbone.return_value = _AnyModel()
    return {
        "distiller": mock_distiller,
        "best_path": best_path,
        "best_loss": 0.5,
        "best_recall": 0.8,
        "compression_ratio": 2.0,
        "elapsed": 1.0,
        "student_model": "efficientvit_b1",
        "student_dim": 384,
        "teacher_dim": 384,
    }


# ── DistillConfig ─────────────────────────────────────────────────────────────


class TestDistillConfigStageField(unittest.TestCase):
    def test_stage_defaults_to_1(self):
        from selfsuvis.pipeline.training.distill import DistillConfig

        self.assertEqual(DistillConfig().stage, 1)

    def test_stage_can_be_set_to_2(self):
        from selfsuvis.pipeline.training.distill import DistillConfig

        self.assertEqual(DistillConfig(stage=2).stage, 2)

    def test_lambda_caption_anchor_defaults_zero(self):
        from selfsuvis.pipeline.training.distill import DistillConfig

        self.assertEqual(DistillConfig().lambda_caption_anchor, 0.0)

    def test_lambda_caption_anchor_is_settable(self):
        from selfsuvis.pipeline.training.distill import DistillConfig

        self.assertEqual(DistillConfig(lambda_caption_anchor=0.5).lambda_caption_anchor, 0.5)


# ── KnowledgeDistiller._load_student ─────────────────────────────────────────


class TestLoadStudentEfficientViT(unittest.TestCase):
    def _distiller_shell(self, student_model: str):
        from selfsuvis.pipeline.training.distill import DistillConfig, KnowledgeDistiller

        d = KnowledgeDistiller.__new__(KnowledgeDistiller)
        d.config = DistillConfig(device="cpu", student_model=student_model)
        d.device = "cpu"
        return d

    def test_efficientvit_branch_calls_embedder(self):
        mock_backbone = _AnyModel()
        mock_inst = MagicMock()
        mock_inst.as_torch_backbone.return_value = mock_backbone
        mock_cls = MagicMock(return_value=mock_inst)
        fake_mod = types.ModuleType("selfsuvis.models.efficientvit_model")
        fake_mod.EfficientViTEmbedder = mock_cls

        with patch.dict("sys.modules", {"selfsuvis.models.efficientvit_model": fake_mod}):
            d = self._distiller_shell("efficientvit_b1")
            result = d._load_student()

        mock_cls.assert_called_once()
        mock_inst.as_torch_backbone.assert_called_once()
        self.assertIs(result, mock_backbone)

    def test_efficientvit_import_error_raises_with_timm_hint(self):
        with patch.dict("sys.modules", {"selfsuvis.models.efficientvit_model": None}):
            d = self._distiller_shell("efficientvit_b1")
            with self.assertRaises(ImportError) as ctx:
                d._load_student()
        self.assertIn("timm", str(ctx.exception))

    def test_dino_model_name_calls_hub_load_dino(self):
        mock_dino = _AnyModel()
        with patch("selfsuvis.models.dino_model.hub_load_dino", return_value=mock_dino) as mock_hub:
            d = self._distiller_shell("dinov2_vits14")
            result = d._load_student()
        mock_hub.assert_called_once_with("dinov2_vits14", pretrained=True)
        self.assertIs(result, mock_dino)


# ── run_distillation_efficientvit ─────────────────────────────────────────────


class TestRunDistillationEfficientViTConfig(unittest.TestCase):
    def test_forces_rkd_a_zero_even_when_caller_sets_nonzero(self):
        """Passing lambda_rkd_a=99 must be silently overridden to 0.0."""
        from selfsuvis.pipeline.training.distill import DistillConfig, run_distillation_efficientvit

        teacher = _AnyModel(out_dim=8)
        bad_cfg = DistillConfig(lambda_rkd_a=99.0, device="cpu")
        captured_cfg = {}

        class _FakeDistiller:
            def __init__(self, teacher_bb, config):
                captured_cfg["cfg"] = config
                # Store real teacher/student so dim inference works
                self.teacher = teacher_bb
                self.student = _AnyModel(out_dim=384)

            def distill(self, frame_paths, ckpt_dir):
                return {
                    "best_path": "",
                    "best_loss": 0.1,
                    "best_recall": 0.9,
                    "compression_ratio": 2.0,
                    "elapsed": 0.1,
                    "student_model": "efficientvit_b1",
                    "student_dim": 384,
                }

            def student_backbone(self):
                return self.student

        with (
            patch("selfsuvis.pipeline.training.distill.KnowledgeDistiller", _FakeDistiller),
            patch("timm.create_model", return_value=_AnyModel(out_dim=384)),
        ):
            run_distillation_efficientvit(teacher, [], Path(tempfile.mkdtemp()), bad_cfg)

        self.assertEqual(captured_cfg["cfg"].lambda_rkd_a, 0.0)
        self.assertEqual(captured_cfg["cfg"].student_model, "efficientvit_b1")

    def test_module_dummy_input_matches_half_precision_weights(self):
        from selfsuvis.pipeline.training.distill import _module_dummy_input

        model = nn.Conv2d(3, 4, kernel_size=1).half()
        dummy = _module_dummy_input(model, image_size=32, device="cpu")

        self.assertEqual(dummy.dtype, torch.float16)


# ── export_efficientvit_onnx ──────────────────────────────────────────────────


class TestExportEfficientViTOnnx(unittest.TestCase):
    def _export(self, backbone, out_path):
        from selfsuvis.pipeline.training.edge_inference import export_efficientvit_onnx

        def _fake_export(model, dummy, path, **kwargs):
            open(path, "wb").close()

        with patch("torch.onnx.export", side_effect=_fake_export) as mock_exp:
            result = export_efficientvit_onnx(backbone, out_path)
        return result, mock_exp

    def test_calls_torch_onnx_export_with_correct_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "ev.onnx")
            _, mock_exp = self._export(_AnyModel(), out_path)
        mock_exp.assert_called_once()
        kw = mock_exp.call_args[1]
        self.assertEqual(kw["input_names"], ["pixel_values"])
        self.assertEqual(kw["output_names"], ["embedding"])
        self.assertEqual(kw["opset_version"], 18)

    def test_returns_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "ev.onnx")
            result, _ = self._export(_AnyModel(), out_path)
        self.assertEqual(result, out_path)

    def test_creates_missing_parent_directory(self):
        from selfsuvis.pipeline.training.edge_inference import export_efficientvit_onnx

        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "new_dir", "ev.onnx")

            def _fake_export(model, dummy, path, **kwargs):
                open(path, "wb").close()

            with patch("torch.onnx.export", side_effect=_fake_export):
                export_efficientvit_onnx(_AnyModel(), nested)

            # Assert inside the temp dir context so the directory still exists
            self.assertTrue(os.path.isdir(os.path.dirname(nested)))


# ── step_distill_stage2 ───────────────────────────────────────────────────────


class TestStepDistillStage2(unittest.TestCase):
    def _run(self, backbone, ckpt_path="", distill_exc=None):
        """Call step_distill_stage2 with standard mocks; return result dict."""
        from selfsuvis.pipeline.workflows.local.steps_distill import step_distill_stage2

        stats = _fake_distiller_stats(ckpt_path) if distill_exc is None else None

        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            with (
                patch(
                    "selfsuvis.pipeline.training.distill.run_distillation_efficientvit",
                    return_value=stats,
                    side_effect=distill_exc,
                ),
                patch(
                    "selfsuvis.pipeline.training.edge_inference.export_efficientvit_onnx",
                    return_value=str(video_dir / "edge_models" / "efficientvit_local.onnx"),
                ),
                patch("selfsuvis.pipeline.workflows.local.steps_report.write_distill_stats_md"),
            ):
                result = step_distill_stage2(
                    backbone,
                    [("f.jpg", 0.0)],
                    "vid",
                    video_dir,
                    "cpu",
                    distill_epochs=1,
                    batch_size=4,
                )
        return result

    def _run_capturing_config(self, backbone, ckpt_path):
        """Return (result, captured_config) to check what DistillConfig was passed."""
        from selfsuvis.pipeline.workflows.local.steps_distill import step_distill_stage2

        stats = _fake_distiller_stats(ckpt_path)
        captured = {}

        def _fake_run(teacher_bb, frame_paths, ckpt_dir, config=None):
            captured["cfg"] = config
            return stats

        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            with (
                patch(
                    "selfsuvis.pipeline.training.distill.run_distillation_efficientvit",
                    side_effect=_fake_run,
                ),
                patch(
                    "selfsuvis.pipeline.training.edge_inference.export_efficientvit_onnx",
                    return_value=str(video_dir / "edge_models" / "efficientvit_local.onnx"),
                ),
                patch("selfsuvis.pipeline.workflows.local.steps_report.write_distill_stats_md"),
            ):
                result = step_distill_stage2(
                    backbone,
                    [("f.jpg", 0.0)],
                    "vid",
                    video_dir,
                    "cpu",
                    distill_epochs=1,
                    batch_size=4,
                )
        return result, captured.get("cfg")

    def test_skips_when_backbone_is_none(self):
        from selfsuvis.pipeline.workflows.local.steps_distill import step_distill_stage2

        with tempfile.TemporaryDirectory() as tmp:
            result = step_distill_stage2(
                None, [], "vid", Path(tmp), "cpu", distill_epochs=1, batch_size=4
            )
        self.assertTrue(result["skipped"])
        self.assertIsNone(result["student_backbone"])

    def test_skips_when_distillation_raises(self):
        result = self._run(_AnyModel(), distill_exc=RuntimeError("OOM"))
        self.assertTrue(result["skipped"])

    def test_skips_when_best_path_empty(self):
        result = self._run(_AnyModel(), ckpt_path="")
        self.assertTrue(result["skipped"])

    def test_success_not_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            result = self._run(_AnyModel(), ckpt_path=ckpt_path)
            self.assertFalse(result["skipped"])
        finally:
            os.unlink(ckpt_path)

    def test_success_returns_student_backbone(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            result = self._run(_AnyModel(), ckpt_path=ckpt_path)
            self.assertIsNotNone(result["student_backbone"])
        finally:
            os.unlink(ckpt_path)

    def test_success_ckpt_mb_positive(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"x" * 2048)
            ckpt_path = f.name
        try:
            result = self._run(_AnyModel(), ckpt_path=ckpt_path)
            self.assertGreater(result["ckpt_mb"], 0.0)
        finally:
            os.unlink(ckpt_path)

    def test_passes_stage_2_and_rkd_a_zero(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _, cfg = self._run_capturing_config(_AnyModel(), ckpt_path=ckpt_path)
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg.stage, 2)
            self.assertEqual(cfg.lambda_rkd_a, 0.0)
            self.assertEqual(cfg.student_model, "efficientvit_b1")
        finally:
            os.unlink(ckpt_path)

    def test_onnx_export_failure_not_skipped(self):
        """ONNX failure must leave skipped=False and onnx_exported=False."""
        from selfsuvis.pipeline.workflows.local.steps_distill import step_distill_stage2

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            stats = _fake_distiller_stats(ckpt_path)
            with tempfile.TemporaryDirectory() as tmp:
                with (
                    patch(
                        "selfsuvis.pipeline.training.distill.run_distillation_efficientvit",
                        return_value=stats,
                    ),
                    patch(
                        "selfsuvis.pipeline.training.edge_inference.export_efficientvit_onnx",
                        side_effect=RuntimeError("ONNX failed"),
                    ),
                    patch("selfsuvis.pipeline.workflows.local.steps_report.write_distill_stats_md"),
                ):
                    result = step_distill_stage2(
                        _AnyModel(),
                        [("f.jpg", 0.0)],
                        "vid",
                        Path(tmp),
                        "cpu",
                        distill_epochs=1,
                        batch_size=4,
                    )
            self.assertFalse(result["skipped"])
            self.assertFalse(result["onnx_exported"])
            self.assertEqual(result["onnx_path"], "")
        finally:
            os.unlink(ckpt_path)


if __name__ == "__main__":
    unittest.main()
