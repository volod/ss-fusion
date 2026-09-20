"""Model-artifact manifests and the file helpers shared with mission bundles.

A manifest is a JSON document shaped by an ss-common contract: `model-artifact` for one model
file (this module) and `mission-bundle` for a recorded mission
(`selfsuvis.pipeline.media.mission_bundle`). Builders return plain dicts in the canonical wire
form the contracts define (UTC timestamps with `Z`, unset optional fields omitted), so the
generated `ss_contracts` models validate them and round-trip them unchanged. Paths inside a
manifest are POSIX paths relative to the directory that holds the manifest.
"""

import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_MANIFEST_SUFFIX = ".manifest.json"

# Model file suffix -> the contract's `format` value.
_MODEL_FORMATS: dict[str, str] = {".onnx": "onnx", ".pt": "pytorch", ".pth": "pytorch"}

_DIGEST_CHUNK = 1 << 20


def file_digest(path: str | Path) -> tuple[str, int]:
    """SHA-256 (lowercase hex) and size of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def relative_manifest_path(path: str | Path, manifest_dir: str | Path) -> str:
    """POSIX path of `path` relative to `manifest_dir`; files outside it are rejected."""
    resolved = Path(path).resolve()
    root = Path(manifest_dir).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{resolved} is outside the manifest directory {root}") from exc
    if any(part.startswith(".") for part in relative.parts):
        raise ValueError(f"{relative} has a path segment starting with a dot")
    return relative.as_posix()


def parent_relative_path(path: str | Path, manifest_dir: str | Path) -> str:
    """POSIX path of `path` relative to `manifest_dir`, climbing out with `../` when needed."""
    relative = Path(os.path.relpath(Path(path).resolve(), Path(manifest_dir).resolve()))
    inner = [part for part in relative.parts if part != ".."]
    if any(part.startswith(".") for part in inner):
        raise ValueError(f"{relative} has a path segment starting with a dot")
    return relative.as_posix()


def utc_timestamp(value: datetime) -> str:
    """An aware datetime in the contracts' canonical form (UTC, `Z`, no zero microseconds)."""
    if value.tzinfo is None:
        raise ValueError("manifest timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def file_mtime(path: str | Path) -> datetime:
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=UTC)


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write a manifest as indented JSON and return its path."""
    target = Path(path)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def model_manifest_path(model_path: str | Path) -> Path:
    """Where a model file's manifest lives: `<file>.manifest.json` next to it."""
    model = Path(model_path)
    return model.with_name(model.name + MODEL_MANIFEST_SUFFIX)


def _metrics(values: Mapping[str, float] | None) -> list[dict[str, Any]]:
    # JSON has no NaN or infinity; a metric the producer could not compute is left out.
    return [
        {"name": name, "value": float(value)}
        for name, value in (values or {}).items()
        if value is not None and math.isfinite(float(value))
    ]


def build_model_artifact(
    model_path: str | Path,
    *,
    base_model: str,
    producer: str,
    artifact_id: str | None = None,
    manifest_dir: str | Path | None = None,
    created_at: datetime | None = None,
    derived_from: str | Path | None = None,
    training_data: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    image_size: int | None = None,
    opset: int | None = None,
    quantization: str | None = None,
) -> dict[str, Any]:
    """A `model-artifact` manifest for one PyTorch checkpoint or ONNX file.

    `manifest_dir` defaults to the model's directory, `artifact_id` to the file stem, and
    `created_at` to the file's modification time. `derived_from` names the checkpoint an export
    was made from; `training_data` holds `kind`, `ref`, and optionally `sample_count`.
    """
    model = Path(model_path)
    fmt = _MODEL_FORMATS.get(model.suffix.lower())
    if fmt is None:
        raise ValueError(f"{model.name}: not a model file {sorted(_MODEL_FORMATS)}")
    root = Path(manifest_dir) if manifest_dir is not None else model.parent
    sha256, size = file_digest(model)
    manifest: dict[str, Any] = {
        "artifact_id": artifact_id or model.stem,
        "format": fmt,
        "path": relative_manifest_path(model, root),
        "sha256": sha256,
        "size_bytes": size,
        "base_model": base_model,
        "producer": producer,
        "created_at": utc_timestamp(created_at or file_mtime(model)),
    }
    if derived_from is not None:
        manifest["derived_from"] = {
            "path": parent_relative_path(derived_from, root),
            "sha256": file_digest(derived_from)[0],
        }
    if training_data:
        manifest["training_data"] = {k: v for k, v in training_data.items() if v is not None}
    manifest["metrics"] = _metrics(metrics)
    optional = {"image_size": image_size, "opset": opset, "quantization": quantization}
    manifest.update((key, value) for key, value in optional.items() if value is not None)
    return manifest


def finetune_model_artifact(
    checkpoint_path: str | Path,
    *,
    model_version_id: str,
    base_model: str,
    result: Mapping[str, Any],
    annotation_count: int,
    created_at: datetime,
    cvat_xml_path: str | None = None,
) -> dict[str, Any]:
    """The manifest of an accepted supervised fine-tuning checkpoint.

    Mirrors the `model_checkpoints` row the FINETUNE handler registers: the model version id,
    the annotated-frame watermark, and the accuracy and distribution-shift metrics.
    """
    if cvat_xml_path:
        training_data = {"kind": "cvat_xml", "ref": Path(cvat_xml_path).name}
    else:
        training_data = {"kind": "cvat_annotations", "ref": "frames.al_tag=annotated"}
    training_data["sample_count"] = int(annotation_count)
    return build_model_artifact(
        checkpoint_path,
        base_model=base_model,
        producer="selfsuvis.worker.handlers.finetune",
        artifact_id=model_version_id,
        created_at=created_at,
        training_data=training_data,
        metrics={
            "best_accuracy": result["best_accuracy"],
            "distribution_shift": result.get("distribution_shift", 0.0),
            "epochs": result.get("epochs"),
        },
    )
