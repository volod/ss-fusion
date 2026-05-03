"""Unit tests for NaN guard in KnowledgeDistiller._forward_teacher.

Ensures that all-NaN teacher embeddings (e.g. from Gemma on edge-case inputs)
produce a finite RKD loss rather than NaN/inf that would corrupt training.
"""

import math

import torch
import torch.nn as nn


class _AllNaNTeacher(nn.Module):
    """Synthetic teacher that always returns all-NaN embeddings."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], self._dim), float("nan"))


class _ZeroTeacher(nn.Module):
    """Synthetic teacher that always returns all-zero embeddings (edge case)."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self._dim)


def _make_distiller(teacher: nn.Module):
    """Create a KnowledgeDistiller with a tiny student, no hub download."""
    from selfsuvis.pipeline.training.distill import DistillConfig, KnowledgeDistiller

    # Tiny linear student avoids DINOv3 hub download in unit tests
    s_dim = 4
    t_dim = 8

    class _TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, s_dim)

        def forward(self, x):
            return self.fc(x.view(x.shape[0], -1))

    cfg = DistillConfig(
        device="cpu",
        epochs=1,
        batch_size=4,
        lambda_rkd_d=25.0,
        lambda_rkd_a=50.0,
        lambda_kd=1.0,
        lambda_koleo=0.1,
    )
    distiller = KnowledgeDistiller.__new__(KnowledgeDistiller)
    distiller.config = cfg
    distiller.device = "cpu"
    distiller.teacher = teacher.eval()
    for p in distiller.teacher.parameters():
        p.requires_grad_(False)
    distiller._t_dim = t_dim
    distiller.student = _TinyStudent()
    distiller._s_dim = s_dim
    distiller._proj = nn.Linear(s_dim, t_dim, bias=False)
    nn.init.orthogonal_(distiller._proj.weight)
    distiller._teacher_params = 0
    distiller._student_params = sum(p.numel() for p in distiller.student.parameters())
    return distiller


def test_all_nan_teacher_produces_finite_embedding():
    """_forward_teacher replaces NaN with 0 and returns a finite, normalised tensor."""

    teacher = _AllNaNTeacher(dim=8)
    distiller = _make_distiller(teacher)

    batch = torch.randn(4, 3, 224, 224)
    t_emb = distiller._forward_teacher(batch)

    assert torch.isfinite(t_emb).all(), "Expected all-finite teacher embedding after NaN guard"


def test_all_nan_teacher_produces_finite_rkd_loss():
    """End-to-end: all-NaN teacher → finite RKD loss (no NaN in training)."""
    from selfsuvis.pipeline.training.distill import rkd_angle_loss, rkd_distance_loss

    teacher = _AllNaNTeacher(dim=8)
    distiller = _make_distiller(teacher)

    batch = torch.randn(4, 3, 224, 224)
    t_emb = distiller._forward_teacher(batch)
    s_proj, _ = distiller._forward_student(batch)

    l_d = rkd_distance_loss(t_emb, s_proj)
    l_a = rkd_angle_loss(t_emb, s_proj)
    total = l_d + l_a

    assert math.isfinite(total.item()), f"Expected finite loss, got {total.item()}"


def test_zero_teacher_embedding_is_handled():
    """All-zero teacher embedding (degenerate) normalises to zero vector without NaN."""
    teacher = _ZeroTeacher(dim=8)
    distiller = _make_distiller(teacher)

    batch = torch.randn(4, 3, 224, 224)
    t_emb = distiller._forward_teacher(batch)

    # Zero vectors normalise to zero — not NaN
    assert torch.isfinite(t_emb).all(), "Zero teacher should produce finite (zero) embedding"


def test_best_checkpoint_prefers_recall_over_loss():
    from selfsuvis.pipeline.training.distill import _should_update_best_checkpoint

    assert _should_update_best_checkpoint(
        best_recall=0.40,
        best_loss=1.0,
        candidate_recall=0.55,
        candidate_loss=1.8,
    ) is True
    assert _should_update_best_checkpoint(
        best_recall=0.55,
        best_loss=1.0,
        candidate_recall=0.55,
        candidate_loss=0.8,
    ) is True
    assert _should_update_best_checkpoint(
        best_recall=0.55,
        best_loss=0.8,
        candidate_recall=0.50,
        candidate_loss=0.1,
    ) is False
