"""Supervised contrastive fine-tuning of DINOv3 on CVAT-annotated mission frames.

Uses the Supervised Contrastive Loss (SupCon, Khosla et al. NeurIPS 2020) to pull
together embeddings of frames sharing the same annotation label and push apart
embeddings from different classes.

Complements self-supervised fine-tuning (ssl_finetune.py): run SSL first to
adapt the backbone to the mission domain, then run supervised fine-tuning on the
smaller annotated subset to specialise class representations.

Input:
  - CVAT XML 1.1 annotation file exported from CVAT
  - Frames directory (JPEGs/PNGs named to match CVAT XML image entries)

Output:
  {SUP_CHECKPOINT_DIR}/dino_sup_{epoch:03d}.pt  — per-epoch checkpoints (backbone only)
  {SUP_CHECKPOINT_DIR}/dino_sup_best.pt          — best (lowest loss) checkpoint

Loading the fine-tuned model:
  Set DINO_CHECKPOINT=/path/to/dino_sup_best.pt before starting the worker/API.
  DINOEmbedder will load the weights automatically.

Label convention:
  The CVAT XML <box label="..."> attribute names map to integer indices in
  alphabetical order of their first occurrence in the XML <labels> block.
  Only the majority label per image is used when an image has multiple objects.

Usage (standalone):
    python scripts/supervised_finetune_dino.py \\
        --frames-dir data/frames \\
        --cvat-xml data/cvat_annotations.xml \\
        --output-dir data/checkpoints/supervised
"""
import glob
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# VisDrone-2019-inspired default label vocabulary (alphabetical = index order)
VISDRONE_LABELS: List[str] = [
    "bicycle",
    "bus",
    "car",
    "motor",
    "pedestrian",
    "truck",
]


# ── Label taxonomy normalization ──────────────────────────────────────────────

def _normalize_labels(
    rows: List[Tuple[str, str]],
    mappings: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Apply a canonical label mapping to (frame_path, label) rows.

    Labels absent from *mappings* are passed through unchanged.
    Logs a warning for each frame where different annotation campaigns assigned
    conflicting labels after normalization (frame appears more than once with
    different labels).

    Args:
        rows:     List of (frame_path, raw_label) tuples.
        mappings: Dict mapping raw label → canonical label (from CVAT_LABEL_MAPPINGS).

    Returns:
        Deduplicated list of (frame_path, canonical_label) tuples.
        When a frame has conflicting labels after normalization, the
        alphabetically first canonical label is kept.
    """
    if not mappings and not rows:
        return rows

    normalized: List[Tuple[str, str]] = []
    seen: Dict[str, str] = {}  # frame_path → canonical label seen so far

    for frame_path, raw_label in rows:
        canonical = mappings.get(raw_label, raw_label)
        if frame_path in seen:
            prev = seen[frame_path]
            if prev != canonical:
                logger.warning(
                    "Label conflict for frame %s: '%s' vs '%s' (after normalization) "
                    "— keeping '%s' (alphabetically first)",
                    frame_path, prev, canonical, min(prev, canonical),
                )
                # Keep alphabetically first for determinism
                if canonical < prev:
                    seen[frame_path] = canonical
        else:
            seen[frame_path] = canonical

    return [(fp, lbl) for fp, lbl in seen.items()]


# ── CVAT XML parser ────────────────────────────────────────────────────────────

class CvatAnnotationParser:
    """Parse a CVAT XML 1.1 annotation file.

    Extracts a mapping from image basename → label string.
    When an image has multiple annotated objects, the label is determined by
    the most frequent (majority-vote) class.  Ties are broken alphabetically.

    Usage:
        parser = CvatAnnotationParser("annotations.xml")
        label_map = parser.frame_labels   # {"frame_0000.jpg": "car", ...}
        labels = parser.label_names       # ["bicycle", "bus", ...] (ordered)
    """

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.label_names: List[str] = []
        self.frame_labels: Dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Extract declared label order from <meta><task><labels>
        label_order: List[str] = []
        for lbl_el in root.findall("./meta/task/labels/label"):
            name_el = lbl_el.find("name")
            if name_el is not None and name_el.text:
                label_order.append(name_el.text.strip())

        # Fallback: collect all labels that appear in box elements
        if not label_order:
            seen: Dict[str, int] = {}
            for img_el in root.findall("image"):
                for box in img_el.findall("box"):
                    lbl = box.get("label", "").strip()
                    if lbl and lbl not in seen:
                        seen[lbl] = len(seen)
            label_order = sorted(seen.keys())

        self.label_names = label_order

        # Parse per-image annotations
        for img_el in root.findall("image"):
            name_attr = img_el.get("name", "")
            basename = os.path.basename(name_attr)
            if not basename:
                continue

            # Count label occurrences for majority vote
            counts: Dict[str, int] = {}
            for box in img_el.findall("box"):
                lbl = box.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
            for poly in img_el.findall("polygon"):
                lbl = poly.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1
            for pts in img_el.findall("points"):
                lbl = pts.get("label", "").strip()
                if lbl:
                    counts[lbl] = counts.get(lbl, 0) + 1

            if not counts:
                continue

            # Majority vote; alphabetical tiebreak
            majority_label = max(counts, key=lambda k: (counts[k], k))
            self.frame_labels[basename] = majority_label

        logger.info(
            "CvatAnnotationParser: xml=%s labels=%s frames=%d",
            self.xml_path, self.label_names, len(self.frame_labels),
        )

    def label_to_idx(self) -> Dict[str, int]:
        """Return {label_name: int_index} mapping."""
        return {name: i for i, name in enumerate(self.label_names)}


# ── Dataset ────────────────────────────────────────────────────────────────────

def _scan_frames(frames_dir: str) -> Dict[str, str]:
    """Return {basename: abs_path} for all images under frames_dir."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    result: Dict[str, str] = {}
    for ext in exts:
        for p in glob.glob(os.path.join(frames_dir, "**", ext), recursive=True):
            result[os.path.basename(p)] = p
    return result


class AnnotatedFrameDataset(Dataset):
    """Frames with CVAT labels; each item returns two augmented views + label index.

    Intersection of frames found on disk and frames annotated (via CVAT XML or DB).
    Frames with labels not present in label_names are skipped.

    Args:
        items:      List of (abs_path, label_idx) tuples.
        transform:  Augmentation transform applied independently to each view.
        two_views:  If True, return (view1, view2, label_idx) for SupCon;
                    if False, return (view1, label_idx) for cross-entropy.
    """

    def __init__(
        self,
        items: List[Tuple[str, int]],
        transform: transforms.Compose,
        two_views: bool = True,
    ):
        self.items = items
        self.transform = transform
        self.two_views = two_views
        if not self.items:
            raise ValueError("No annotated frames provided to AnnotatedFrameDataset.")
        logger.info(
            "AnnotatedFrameDataset: %d annotated frames loaded (two_views=%s)",
            len(self.items), two_views,
        )

    @classmethod
    def from_xml(
        cls,
        frames_dir: str,
        parser: "CvatAnnotationParser",
        transform: transforms.Compose,
        two_views: bool = True,
    ) -> "AnnotatedFrameDataset":
        """Build from a CvatAnnotationParser (CVAT XML path)."""
        from selfsuvis.pipeline.core.config import settings as _settings
        disk_frames = _scan_frames(frames_dir)

        # Apply label taxonomy normalization before building the vocabulary.
        # This ensures labels from different annotation campaigns with
        # different naming conventions map to the same canonical class.
        raw_pairs = [
            (disk_frames[bn], lbl)
            for bn, lbl in parser.frame_labels.items()
            if bn in disk_frames
        ]
        skipped_missing = len(parser.frame_labels) - len(raw_pairs)
        normalized_pairs = _normalize_labels(raw_pairs, _settings.CVAT_LABEL_MAPPINGS)

        all_labels = sorted({lbl for _, lbl in normalized_pairs})
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        items: List[Tuple[str, int]] = []
        skipped_unknown_label = 0

        for frame_path, label_name in normalized_pairs:
            if label_name not in label_to_idx:
                skipped_unknown_label += 1
                continue
            items.append((frame_path, label_to_idx[label_name]))

        if skipped_missing > 0:
            logger.debug(
                "AnnotatedFrameDataset.from_xml: %d frames not found on disk (skipped)",
                skipped_missing,
            )
        if skipped_unknown_label > 0:
            logger.debug(
                "AnnotatedFrameDataset.from_xml: %d frames with unknown label (skipped)",
                skipped_unknown_label,
            )
        if not items:
            raise ValueError(
                f"No annotated frames found. "
                f"Check that frames_dir={frames_dir!r} contains files named in the CVAT XML."
            )
        return cls(items=items, transform=transform, two_views=two_views)

    @classmethod
    def from_db(
        cls,
        transform: transforms.Compose,
        two_views: bool = True,
        mission_id: Optional[str] = None,
    ) -> "AnnotatedFrameDataset":
        """Build from PostgreSQL frames table (DB-native annotation path).

        Selects frames where al_tag='annotated' and cvat_label IS NOT NULL.
        When mission_id is None, returns annotated frames across all missions.

        Raises ValueError if no annotated frames are found.
        Raises RuntimeError if DATABASE_URL is not configured.
        """
        import asyncio
        import asyncpg
        from selfsuvis.pipeline.core.config import settings

        db_url = settings.DATABASE_URL
        if not db_url:
            raise RuntimeError("DATABASE_URL not configured; cannot load annotated frames from DB")

        async def _fetch():
            conn = await asyncpg.connect(db_url, timeout=10)
            try:
                if mission_id:
                    rows = await conn.fetch(
                        "SELECT frame_path, cvat_label FROM frames "
                        "WHERE al_tag = 'annotated' AND cvat_label IS NOT NULL "
                        "AND mission_id = $1",
                        mission_id,
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT frame_path, cvat_label FROM frames "
                        "WHERE al_tag = 'annotated' AND cvat_label IS NOT NULL"
                    )
                return [(r["frame_path"], r["cvat_label"]) for r in rows]
            finally:
                await conn.close()

        rows = asyncio.run(_fetch())

        if not rows:
            raise ValueError(
                "No annotated frames found in DB "
                f"(mission_id={mission_id!r}). "
                "Annotate frames in CVAT and fire a webhook first."
            )

        # Apply label taxonomy normalization before building the vocabulary.
        raw_pairs = [(r[0], r[1]) for r in rows]
        normalized_pairs = _normalize_labels(raw_pairs, settings.CVAT_LABEL_MAPPINGS)

        # Build label vocabulary from normalized labels (sorted for determinism)
        all_labels = sorted({label for _, label in normalized_pairs})
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        items: List[Tuple[str, int]] = []
        for frame_path, label_name in normalized_pairs:
            if not os.path.isfile(frame_path):
                logger.debug("AnnotatedFrameDataset.from_db: frame not on disk %s", frame_path)
                continue
            items.append((frame_path, label_to_idx[label_name]))

        logger.info(
            "AnnotatedFrameDataset.from_db: %d frames, %d labels (mission_id=%s)",
            len(items), len(all_labels), mission_id,
        )
        return cls(items=items, transform=transform, two_views=two_views)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label_idx = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.two_views:
            return self.transform(img), self.transform(img), label_idx
        return self.transform(img), label_idx


# ── Eval gate ──────────────────────────────────────────────────────────────────

def _stratified_split(
    items: List[Tuple[str, int]],
    eval_fraction: float,
    min_per_class: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split items into (train, eval) lists using stratified sampling.

    Raises ValueError if:
    - fewer than 2 distinct classes (single-class: SupCon cannot train)
    - any class has fewer than min_per_class samples in the eval split
    """
    from collections import defaultdict
    by_class: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)

    if len(by_class) < 2:
        raise ValueError(
            f"Dataset has only {len(by_class)} class(es); SupCon requires ≥2 classes."
        )

    train_items: List[Tuple[str, int]] = []
    eval_items: List[Tuple[str, int]] = []

    for label_idx, class_items in by_class.items():
        shuffled = list(class_items)
        random.shuffle(shuffled)
        n_eval = max(min_per_class, int(len(shuffled) * eval_fraction))
        if n_eval >= len(shuffled):
            # Not enough samples to hold out; skip this class from eval
            train_items.extend(shuffled)
            logger.debug(
                "_stratified_split: class %d has %d samples, skipping eval split",
                label_idx, len(shuffled),
            )
        else:
            eval_items.extend(shuffled[:n_eval])
            train_items.extend(shuffled[n_eval:])

    return train_items, eval_items


def _eval_accuracy(
    backbone: torch.nn.Module,
    eval_items: List[Tuple[str, int]],
    device: str,
    transform: transforms.Compose,
) -> float:
    """Compute 1-NN accuracy on eval_items using backbone embeddings.

    This is a convergence signal (not overfitting detection) for the eval gate.
    Returns a float in [0.0, 1.0].  Returns 0.0 on empty eval set.

    Note: 1-NN accuracy increases monotonically with SupCon training epochs
    because the loss directly clusters same-class embeddings.  It measures
    convergence, not generalisation.  Use _eval_distribution_shift() alongside
    this function to detect potential overfitting.
    """
    if not eval_items:
        return 0.0

    backbone.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for path, label_idx in eval_items:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            t = transform(img).unsqueeze(0).to(device)
            feat = backbone(t)  # (1, D)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            embeddings.append(feat.squeeze(0).cpu())
            labels.append(label_idx)

    if len(embeddings) < 2:
        return 0.0

    E = torch.stack(embeddings)           # (N, D)
    lbl = torch.tensor(labels)            # (N,)

    # Cosine similarity matrix; mask self
    sim = torch.mm(E, E.t())              # (N, N)
    sim.fill_diagonal_(float("-inf"))

    nn_idx = sim.argmax(dim=1)            # (N,)
    correct = (lbl[nn_idx] == lbl).float().mean().item()
    return correct


def _eval_distribution_shift(
    backbone: torch.nn.Module,
    eval_items: List[Tuple[str, int]],
    device: str,
    transform: transforms.Compose,
) -> float:
    """Compute intra-class vs. inter-class cosine similarity gap as an overfitting indicator.

    A SupCon-trained backbone clusters same-class embeddings together (high intra-class
    cosine similarity) and pushes apart different-class embeddings (low inter-class cosine
    similarity).  The gap (intra_mean - inter_mean) measures how tightly the model has
    learned class boundaries.

    Interpretation:
      gap ≈ 0.0  — no class separation (random / underfitting)
      gap ≈ 0.5  — healthy discriminative clustering
      gap > 0.9  — extremely tight clustering; potential memorisation / overfitting

    With 8 frozen transformer blocks and ~500 frames, genuine overfitting is unlikely,
    but the gap is logged as an early warning signal.  It is NOT used as an acceptance
    gate — that remains the 1-NN accuracy threshold.

    Returns the gap in [−2.0, 2.0].  Returns 0.0 if the eval set is too small to compute
    (< 4 samples or < 2 classes).
    """
    if len(eval_items) < 4:
        return 0.0

    # Check at least 2 classes are present
    classes_present = {label for _, label in eval_items}
    if len(classes_present) < 2:
        return 0.0

    backbone.eval()
    embeddings: List[torch.Tensor] = []
    labels: List[int] = []

    with torch.no_grad():
        for path, label_idx in eval_items:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            t = transform(img).unsqueeze(0).to(device)
            feat = backbone(t)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            embeddings.append(feat.squeeze(0).cpu())
            labels.append(label_idx)

    if len(embeddings) < 4:
        return 0.0

    E = torch.stack(embeddings)       # (N, D)
    lbl = torch.tensor(labels)        # (N,)

    sim = torch.mm(E, E.t())          # (N, N) cosine similarities (L2-normalised inputs)
    sim.fill_diagonal_(0.0)           # exclude self-similarity

    same_mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1))  # (N, N) bool
    same_mask.fill_diagonal_(False)
    diff_mask = ~same_mask
    diff_mask.fill_diagonal_(False)

    n_same = same_mask.sum().item()
    n_diff = diff_mask.sum().item()

    if n_same == 0 or n_diff == 0:
        return 0.0

    intra_mean = sim[same_mask].mean().item()
    inter_mean = sim[diff_mask].mean().item()
    gap = intra_mean - inter_mean

    logger.debug(
        "Distribution shift: intra_mean=%.4f inter_mean=%.4f gap=%.4f",
        intra_mean, inter_mean, gap,
    )
    return gap


# ── Loss ───────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. NeurIPS 2020, eq. 2).

    For each anchor i, positives are all other samples sharing the same label.
    The loss is defined only for anchors that have at least one positive.

    When two augmented views are concatenated into a single batch (the standard
    SupCon approach), each view's label appears twice; this naturally includes
    both the same view's pair and same-class samples from other images as positives.

    Args:
        temperature: Softmax temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            z:      (B, D) L2-normalised embeddings.
            labels: (B,)   integer class labels.

        Returns:
            Scalar loss tensor. Returns 0.0 if no anchor has any positive.
        """
        B = z.size(0)
        device = z.device

        # Similarity matrix (B, B), scaled by temperature
        sim = torch.mm(z, z.t()) / self.temperature

        # Positive mask: same label, excluding self-pairs
        labels_col = labels.view(-1, 1)           # (B, 1)
        mask_pos = (labels_col == labels_col.t()).float()  # (B, B)
        mask_self = torch.eye(B, device=device)
        mask_pos = mask_pos - mask_self

        # Anchors with at least one positive
        n_pos = mask_pos.sum(dim=1)               # (B,)
        valid = n_pos > 0

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-denominator: sum over all j ≠ i of exp(sim_ij)
        # Use logsumexp trick for numerical stability
        sim_no_self = sim.masked_fill(mask_self.bool(), float("-inf"))
        log_denom = torch.logsumexp(sim_no_self, dim=1, keepdim=True)  # (B, 1)

        # Log probabilities for each pair
        log_probs = sim - log_denom              # (B, B)

        # Mean log-prob over positives per anchor
        loss_per_anchor = -(mask_pos * log_probs).sum(dim=1) / (n_pos + 1e-8)  # (B,)

        return loss_per_anchor[valid].mean()


# ── Model ──────────────────────────────────────────────────────────────────────

class SupervisedFineTuner:
    """DINOv3/DINOv2 backbone wrapped for supervised contrastive fine-tuning.

    Architecture:
      backbone (frozen first N blocks) → ProjectionHead → L2-normalised vector

    The projection head is discarded at inference; only backbone weights are saved.
    Mirrors DINOFineTuner from ssl_finetune.py but exposes a cleaner interface
    for supervised training.

    Args:
        model_name:    DINOv3 hub model name (e.g. "dinov3_vitb14").
        freeze_blocks: Number of transformer blocks to freeze from the start.
        device:        torch device string.
        embed_dim:     Backbone output dimension (768 for ViT-B, 1024 for ViT-L).
        proj_out_dim:  Projection head output dimension.
        ssl_checkpoint: Optional path to a prior SSL backbone checkpoint to load
                        before fine-tuning (domain-adapted starting point).
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb14",
        freeze_blocks: int = 8,
        device: str = "cpu",
        embed_dim: int = 768,
        proj_out_dim: int = 128,
        ssl_checkpoint: Optional[str] = None,
    ):
        self.device = device
        self.model_name = model_name

        from selfsuvis.models.dino_model import hub_load_dino
        self.backbone = hub_load_dino(model_name, pretrained=True)
        self.backbone = self.backbone.to(device)

        if ssl_checkpoint and os.path.isfile(ssl_checkpoint):
            state = torch.load(ssl_checkpoint, map_location=device)
            self.backbone.load_state_dict(state)
            logger.info("SupervisedFineTuner: loaded SSL checkpoint %s", ssl_checkpoint)

        self._freeze_blocks(freeze_blocks)

        # Two-layer MLP projection head (same design as ssl_finetune.ProjectionHead)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_out_dim),
        ).to(device)

        trainable = sum(p.numel() for p in self._all_params() if p.requires_grad)
        logger.info(
            "SupervisedFineTuner: model=%s freeze_blocks=%d trainable_params=%d",
            model_name, freeze_blocks, trainable,
        )

    def _freeze_blocks(self, n: int) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        blocks = list(self.backbone.blocks)
        for block in blocks[n:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def _all_params(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def trainable_params(self):
        return [p for p in self._all_params() if p.requires_grad]

    def train(self) -> None:
        self.backbone.train()
        self.head.train()

    def eval(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """backbone CLS token → projection head → L2-normalised vector."""
        feats = self.backbone(x)          # (B, embed_dim)
        proj = self.head(feats)           # (B, proj_out_dim)
        return F.normalize(proj, dim=-1)  # L2-normalised

    def save_checkpoint(self, path: str) -> None:
        """Save backbone state dict only (head discarded at inference)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.backbone.state_dict(), path)
        logger.info("Checkpoint saved: %s", path)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class SupervisedFinetuneConfig:
    frames_dir: str
    output_dir: str
    cvat_xml_path: Optional[str] = None  # None → use from_db() path
    model_name: str = "dinov3_vitb14"
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.04
    temperature: float = 0.07
    freeze_blocks: int = 8       # fewer frozen blocks than SSL (more fine-tuning freedom)
    embed_dim: int = 768
    proj_out_dim: int = 128
    num_workers: int = 4
    save_every: int = 1
    device: str = "cpu"
    seed: int = 42
    ssl_checkpoint: Optional[str] = None   # warm-start from SSL backbone if set
    eval_fraction: float = 0.1             # fraction of data held out for eval gate
    min_per_class_eval: int = 2            # min eval samples per class
    overfitting_shift_threshold: float = 0.9  # gap > this logs an overfitting warning
    min_eval_gate_frames: int = 20         # reject dataset smaller than this
    eval_gate_threshold: float = 0.6       # min 1-NN accuracy to accept checkpoint
    mission_id: Optional[str] = None       # when using from_db(), filter by mission


# ── Training loop ──────────────────────────────────────────────────────────────

def run_supervised_finetune(cfg: SupervisedFinetuneConfig) -> Dict[str, Any]:
    """Run supervised contrastive fine-tuning on CVAT-annotated frames.

    Args:
        cfg: SupervisedFinetuneConfig instance.

    Returns:
        Dict with keys:
            path               — str, path to best checkpoint (or "" if not accepted)
            best_accuracy      — float, 1-NN eval accuracy on held-out split (convergence signal)
            epochs             — int, number of training epochs completed
            accepted           — bool, True if checkpoint passed the eval gate
            distribution_shift — float, intra-class vs. inter-class cosine similarity gap;
                                  values > overfitting_shift_threshold indicate potential
                                  memorisation (logged as warning, not used as gate).
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    transform = _build_augment_transform()

    # Build dataset — XML path or DB-native
    if cfg.cvat_xml_path:
        parser = CvatAnnotationParser(cfg.cvat_xml_path)
        if not parser.label_names:
            raise ValueError(f"No labels found in CVAT XML: {cfg.cvat_xml_path}")
        logger.info(
            "Supervised fine-tuning (XML): labels=%s total_annotated=%d",
            parser.label_names, len(parser.frame_labels),
        )
        full_dataset = AnnotatedFrameDataset.from_xml(cfg.frames_dir, parser, transform, two_views=True)
    else:
        full_dataset = AnnotatedFrameDataset.from_db(transform, two_views=True, mission_id=cfg.mission_id)

    all_items = full_dataset.items

    # Reject datasets that are too small to run the eval gate
    if len(all_items) < cfg.min_eval_gate_frames:
        logger.warning(
            "Dataset too small for eval gate: %d frames < min=%d. Rejecting.",
            len(all_items), cfg.min_eval_gate_frames,
        )
        return {"path": "", "best_accuracy": 0.0, "epochs": 0, "accepted": False, "distribution_shift": 0.0}

    # Stratified train/eval split (raises ValueError on single-class dataset)
    try:
        train_items, eval_items = _stratified_split(
            all_items,
            eval_fraction=cfg.eval_fraction,
            min_per_class=cfg.min_per_class_eval,
        )
    except ValueError as exc:
        logger.warning("Stratified split failed: %s. Rejecting dataset.", exc)
        return {"path": "", "best_accuracy": 0.0, "epochs": 0, "accepted": False, "distribution_shift": 0.0}

    logger.info(
        "Dataset: %d train | %d eval | epochs=%d | batch=%d | device=%s",
        len(train_items), len(eval_items), cfg.epochs, cfg.batch_size, cfg.device,
    )

    train_dataset = AnnotatedFrameDataset(items=train_items, transform=transform, two_views=True)

    # SupCon needs ≥2 samples per class per batch to form positives.
    # Drop last incomplete batch so SupConLoss always has full batches.
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
        drop_last=True,
    )

    # Model + optimiser
    tuner = SupervisedFineTuner(
        model_name=cfg.model_name,
        freeze_blocks=cfg.freeze_blocks,
        device=cfg.device,
        embed_dim=cfg.embed_dim,
        proj_out_dim=cfg.proj_out_dim,
        ssl_checkpoint=cfg.ssl_checkpoint,
    )
    optimizer = torch.optim.AdamW(
        tuner.trainable_params(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    loss_fn = SupConLoss(temperature=cfg.temperature)

    best_loss = float("inf")
    best_path = os.path.join(cfg.output_dir, "dino_sup_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        tuner.train()
        epoch_losses: List[float] = []

        for v1, v2, labels in loader:
            v1 = v1.to(cfg.device)
            v2 = v2.to(cfg.device)
            labels = labels.to(cfg.device)

            # Two views per sample → concatenate for SupCon
            # Shape: (2B, proj_out_dim) with labels repeated
            z1 = tuner.forward(v1)
            z2 = tuner.forward(v2)
            z = torch.cat([z1, z2], dim=0)                     # (2B, D)
            labels_2x = torch.cat([labels, labels], dim=0)     # (2B,)

            loss = loss_fn(z, labels_2x)

            # Skip optimizer step when no positives exist in this batch (loss=0.0)
            if not loss.requires_grad or loss.item() == 0.0:
                logger.debug("Batch has no positives — skipping optimizer step")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        logger.info(
            "Epoch %d/%d  loss=%.4f  lr=%.2e",
            epoch, cfg.epochs, avg_loss, scheduler.get_last_lr()[0],
        )

        if epoch % cfg.save_every == 0:
            ckpt = os.path.join(cfg.output_dir, f"dino_sup_{epoch:03d}.pt")
            tuner.save_checkpoint(ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            tuner.save_checkpoint(best_path)
            logger.info("New best checkpoint: loss=%.4f → %s", best_loss, best_path)

    # Eval gate: 1-NN accuracy (convergence signal) on held-out split
    best_accuracy = _eval_accuracy(tuner.backbone, eval_items, cfg.device, transform)
    accepted = best_accuracy >= cfg.eval_gate_threshold

    # Overfitting indicator: intra-class vs. inter-class cosine similarity gap.
    # Not used as a gate — logged as a warning when the gap is suspiciously high.
    distribution_shift = _eval_distribution_shift(tuner.backbone, eval_items, cfg.device, transform)
    if distribution_shift > cfg.overfitting_shift_threshold:
        logger.warning(
            "Potential overfitting: distribution_shift=%.4f > threshold=%.2f  "
            "(intra-class clustering is very tight — may indicate memorisation on small dataset)",
            distribution_shift, cfg.overfitting_shift_threshold,
        )

    logger.info(
        "Supervised fine-tuning complete. best_loss=%.4f  eval_accuracy=%.4f  "
        "gate=%.2f  accepted=%s  distribution_shift=%.4f  checkpoint=%s",
        best_loss, best_accuracy, cfg.eval_gate_threshold, accepted,
        distribution_shift, best_path,
    )
    return {
        "path": best_path if accepted else "",
        "best_accuracy": best_accuracy,
        "epochs": cfg.epochs,
        "accepted": accepted,
        "distribution_shift": distribution_shift,
    }


# ── Augmentation (shared with ssl_finetune) ────────────────────────────────────

def _build_augment_transform(image_size: int = 224) -> transforms.Compose:
    """Strong random augmentation matching SimCLR / ssl_finetune conventions."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Config from environment ────────────────────────────────────────────────────

def config_from_settings(
    frames_dir: str,
    cvat_xml_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    mission_id: Optional[str] = None,
) -> SupervisedFinetuneConfig:
    """Build SupervisedFinetuneConfig from pipeline.core.config.settings.

    Args:
        frames_dir:    Path to annotated frames directory (used only with cvat_xml_path).
        cvat_xml_path: Path to CVAT XML annotation file.  When None, uses from_db() path.
        output_dir:    Override checkpoint output directory (default: settings.SUP_CHECKPOINT_DIR).
        mission_id:    Filter frames by mission when using the DB path.
    """
    from selfsuvis.pipeline.core.config import settings

    device = settings.DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "dinov3_vitb14" if settings.MODEL_NAME == "dinov3" else "dinov2_vitb14"
    embed_dim = 1024 if "vitl" in model_name else 768

    ssl_ckpt = settings.DINO_CHECKPOINT if settings.DINO_CHECKPOINT else None

    return SupervisedFinetuneConfig(
        frames_dir=frames_dir,
        cvat_xml_path=cvat_xml_path,
        output_dir=output_dir or settings.SUP_CHECKPOINT_DIR,
        model_name=model_name,
        epochs=settings.SUP_FINETUNE_EPOCHS,
        batch_size=settings.SUP_FINETUNE_BATCH_SIZE,
        lr=settings.SUP_FINETUNE_LR,
        temperature=settings.SUP_FINETUNE_TEMPERATURE,
        freeze_blocks=settings.SUP_FINETUNE_FREEZE_BLOCKS,
        embed_dim=embed_dim,
        device=device,
        ssl_checkpoint=ssl_ckpt,
        eval_fraction=settings.SUP_EVAL_FRACTION,
        min_per_class_eval=settings.SUP_MIN_PER_CLASS_EVAL,
        min_eval_gate_frames=settings.SUP_MIN_EVAL_GATE_FRAMES,
        eval_gate_threshold=settings.SUP_EVAL_GATE_THRESHOLD,
        overfitting_shift_threshold=settings.SUP_OVERFITTING_SHIFT_THRESHOLD,
        mission_id=mission_id,
    )
