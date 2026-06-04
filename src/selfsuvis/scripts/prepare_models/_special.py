"""Specialised downloaders for YOLO, SceneTok, and SAM (with their cache checkers)."""

import contextlib
import io
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from selfsuvis.pipeline.core.logging import get_logger

from ._utils import _CACHE_DIR, _importable_module

log = get_logger("prepare_models")

# -- YOLO ----------------------------------------------------------------------

_YOLO_CACHE_DIR = _CACHE_DIR / "ultralytics"


def _is_yolo_cached(model_id: str) -> bool:
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    return (_YOLO_CACHE_DIR / model_file).exists()


def _download_yolo(model_id: str) -> None:
    """Download YOLO11 weights via ultralytics auto-download.

    Weights are stored in ``.data/.cache/ultralytics/`` — the full path is passed
    to the YOLO constructor so ultralytics downloads there instead of cwd.
    """
    model_file = model_id if model_id.endswith(".pt") else f"{model_id}.pt"
    log.info("YOLO11 — model=%s", model_file)

    ult_cache = _YOLO_CACHE_DIR / model_file
    if ult_cache.exists():
        log.info("  [ok] YOLO11 already cached at %s — skipping download", ult_cache)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning(
            "  ultralytics not installed — skipping YOLO download (pip install ultralytics)"
        )
        return

    _YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        import numpy as np

        os.environ["YOLO_VERBOSE"] = "False"
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            model = YOLO(str(ult_cache))
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            model(dummy, verbose=False)
        log.info("  [ok] YOLO11 ready  (%.1fs)  file=%s", time.monotonic() - t0, ult_cache)
    except Exception as exc:
        log.warning("  YOLO11 download/verification failed: %s", exc)
        raise


# -- SceneTok ------------------------------------------------------------------

_SCENETOK_GITHUB_URL = "https://github.com/mohammadasim98/scenetok"
_SCENETOK_HF_DEPS = [
    "hustvl/vavae-imagenet256-f16d32-dinov2",
    "hpcai-tech/Open-Sora-v2-Video-DC-AE",
]
_SCENETOK_DEFAULT_CHECKPOINT = "va-videodc_re10k"
_SCENETOK_CHECKPOINT_VARIANTS = ["va-videodc_re10k", "va-videodc_dl3dv", "va-wan_dl3dv"]
_SCENETOK_CACHE_DIR = _CACHE_DIR / "selfsuvis" / "scenetok"
_SCENETOK_CHECKPOINT_URLS: dict = {
    "va-videodc_re10k": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/6Y7EsosfbnpcRxj/download",
    "va-videodc_dl3dv": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/aYBX7atFNKkmdSE/download",
    "va-wan_dl3dv": "https://nextcloud.mpi-klsb.mpg.de/index.php/s/X7yzk7QANtwawPc/download",
}


def _normalize_scenetok_checkpoint_name(checkpoint_name: str) -> str:
    raw = (checkpoint_name or "").strip()
    if not raw:
        raise ValueError(
            "SceneTok checkpoint name is empty. "
            f"Known checkpoints: {', '.join(_SCENETOK_CHECKPOINT_VARIANTS)}"
        )
    ckpt_key = raw[:-5] if raw.endswith(".ckpt") else raw
    if ckpt_key not in _SCENETOK_CHECKPOINT_URLS:
        known = ", ".join(_SCENETOK_CHECKPOINT_VARIANTS)
        raise ValueError(
            f"Unknown SceneTok checkpoint {checkpoint_name!r}. Known checkpoints: {known}"
        )
    return f"{ckpt_key}.ckpt"


def _is_scenetok_cached(checkpoint_name: str) -> bool:
    ckpt_file = _normalize_scenetok_checkpoint_name(checkpoint_name)
    return (_SCENETOK_CACHE_DIR / ckpt_file).exists()


def _download_scenetok(checkpoint_name: str) -> None:
    """Install the scenetok package, download HF dependencies, and cache the checkpoint.

    Checkpoint is stored at ``.data/.cache/selfsuvis/scenetok/<name>.ckpt`` — the
    same path that ``scenetok_server.py:_checkpoint_path()`` resolves at runtime.
    """
    ckpt_file = _normalize_scenetok_checkpoint_name(checkpoint_name)
    ckpt_path = _SCENETOK_CACHE_DIR / ckpt_file
    log.info("SceneTok — checkpoint=%s", checkpoint_name)

    src_dir = _SCENETOK_CACHE_DIR.parent / "scenetok_src"
    _scenetok_installed = _importable_module("scenetok")
    if not _scenetok_installed:
        t0 = time.monotonic()
        if not src_dir.exists():
            log.info("  scenetok package not found — cloning from %s …", _SCENETOK_GITHUB_URL)
            src_dir.mkdir(parents=True, exist_ok=True)
            clone_result = subprocess.run(
                ["git", "clone", "--depth", "1", _SCENETOK_GITHUB_URL, str(src_dir)],
                check=False,
            )
            if clone_result.returncode != 0:
                shutil.rmtree(src_dir, ignore_errors=True)
                raise RuntimeError(
                    f"git clone failed (exit {clone_result.returncode}).\n"
                    "  Ensure git is installed and github.com is reachable."
                )
        else:
            log.info("  scenetok source already cloned at %s — skipping clone", src_dir)

        if not (src_dir / "setup.py").exists() and not (src_dir / "pyproject.toml").exists():
            if (src_dir / "src" / "scenetok").is_dir():
                pkg_spec = "package_dir={'': 'src'}, packages=find_packages(where='src')"
            else:
                pkg_spec = "packages=find_packages()"
            (src_dir / "setup.py").write_text(
                "from setuptools import setup, find_packages\n"
                f"setup(name='scenetok', version='0.1.0', {pkg_spec})\n"
            )
            log.info("  synthesised setup.py for unpackaged repo")

        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(src_dir), "-q"],
            check=False,
        )
        if install_result.returncode != 0:
            raise RuntimeError(
                f"pip install -e {src_dir} failed (exit {install_result.returncode}).\n"
                f"  Manual fix: pip install -e {src_dir}"
            )
        if _importable_module("scenetok"):
            log.info(
                "  [ok] scenetok package installed  (%.1fs)  src=%s", time.monotonic() - t0, src_dir
            )
        else:
            log.warning(
                "  scenetok source was installed as an editable distribution, but `import scenetok` still fails. "
                "Local in-process SceneTok is not ready; use the sidecar mode only after exposing a real importable package."
            )
    else:
        log.info("  [ok] scenetok package already installed")

    for hf_dep in _SCENETOK_HF_DEPS:
        log.info("  HF dep — %s", hf_dep)
        from ._cache import _is_hf_cached

        if _is_hf_cached(hf_dep):
            log.info("  [ok] already cached — %s", hf_dep)
            continue
        t0 = time.monotonic()
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=hf_dep,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        log.info("  [ok] %s  (%.1fs)  cache=%s", hf_dep, time.monotonic() - t0, local_dir)

    if ckpt_path.exists():
        log.info("  [ok] checkpoint already cached — %s", ckpt_path)
        return

    ckpt_key = ckpt_file[:-5]
    url = _SCENETOK_CHECKPOINT_URLS.get(ckpt_key)

    _SCENETOK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(".ckpt.part")
    t0 = time.monotonic()
    log.info("  Downloading %s from MPI Nextcloud …", ckpt_file)
    try:
        _last_log = [0.0]

        def _reporthook(block_num, block_size, total_size):
            now = time.monotonic()
            if now - _last_log[0] >= 30.0:
                _last_log[0] = now
                downloaded_mb = block_num * block_size / 1_048_576
                if total_size > 0:
                    log.info("    %.0f / %.0f MiB", downloaded_mb, total_size / 1_048_576)
                else:
                    log.info("    %.0f MiB downloaded …", downloaded_mb)

        urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
        tmp_path.rename(ckpt_path)
        size_mb = ckpt_path.stat().st_size / 1_048_576
        log.info(
            "  [ok] checkpoint ready  (%.1fs  %.0f MiB)  path=%s",
            time.monotonic() - t0,
            size_mb,
            ckpt_path,
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        log.warning(
            "  Checkpoint download failed: %s\n"
            "  Download %s manually from %s and place at:\n    %s",
            exc,
            ckpt_file,
            _SCENETOK_GITHUB_URL,
            ckpt_path,
        )
        raise


# -- SAM -----------------------------------------------------------------------


def _sam3_accessible() -> bool:
    """Return True only if the SAM3 gated repo files are accessible with the current token.

    model_info() is not sufficient — the model card metadata is public even when
    files are gated.  We probe by attempting to fetch a tiny sentinel file which
    hits the same auth gate as the full snapshot.
    """
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="facebook/sam3",
            filename="model_index.json",
            local_files_only=False,
        )
        return True
    except Exception:
        return False


def _sam3_dialog() -> str:
    """Interactive prompt shown when SAM3 access is not granted.

    Prints instructions, then asks the user what to do next.
    Returns one of: 'retry' | 'sam2' | 'skip'.

    When stdin is not a TTY (CI, piped output) the function returns 'sam2'
    immediately so the setup script never blocks.
    """
    if not sys.stdin.isatty():
        return "sam2"

    print()
    print("  ┌- SAM3 — gated model -------------------------------------------┐")
    print("  │  facebook/sam3 requires HuggingFace access approval.           │")
    print("  │                                                                 │")
    print("  │  To unlock SAM3:                                                │")
    print("  │    1. Visit  https://huggingface.co/facebook/sam3              │")
    print("  │    2. Click 'Access repository' and accept the licence          │")
    print("  │    3. Make sure HF_TOKEN is set in .env (Read scope)            │")
    print("  │    4. Choose [r] Retry below                                    │")
    print("  │                                                                 │")
    print("  │  SAM2 (facebook/sam2-hiera-large) is a fully open fallback.    │")
    print("  └-----------------------------------------------------------------┘")
    print()
    print("  [s]  Use SAM2 fallback  (default)")
    print("  [r]  Retry              (after granting access in another tab)")
    print("  [x]  Skip SAM entirely")
    print()

    while True:
        try:
            raw = input("  Choice [s/r/x]: ")
            raw = raw.encode("ascii", errors="ignore").decode().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "sam2"
        choice = raw or "s"
        if choice in ("s", "sam2"):
            return "sam2"
        if choice in ("r", "retry"):
            return "retry"
        if choice in ("x", "skip"):
            return "skip"
        print("  Please enter s, r, or x.")


def _download_sam(model_id: str) -> None:
    """Download SAM3 (or SAM2 fallback) weights from HuggingFace Hub.

    SAM3 is a gated model.  When access is not granted the function shows an
    interactive dialog (TTY) or auto-continues with SAM2 (non-interactive).
    """
    from ._cache import _is_hf_cached

    SAM2_FALLBACK = "facebook/sam2-hiera-large"
    is_sam3 = "sam3" in model_id.lower()

    if is_sam3 and not _is_hf_cached(model_id):
        while not _sam3_accessible():
            action = _sam3_dialog()
            if action == "retry":
                log.info("  Re-checking SAM3 access …")
                continue
            elif action == "skip":
                log.info("SAM — skipped by user choice.")
                return
            else:
                log.info("SAM — using %s (SAM3 access not granted)", SAM2_FALLBACK)
                model_id = SAM2_FALLBACK
                is_sam3 = False
                break

    log.info("SAM — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] SAM already cached — skipping load")
        return

    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        log.info("  [ok] SAM ready  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        if is_sam3:
            log.info("  SAM3 download failed — falling back to %s", SAM2_FALLBACK)
            if _is_hf_cached(SAM2_FALLBACK):
                log.info("  [ok] SAM2 fallback already cached — skipping download")
                return
            try:
                from huggingface_hub import snapshot_download as _sd

                local_dir = _sd(
                    repo_id=SAM2_FALLBACK,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
                )
                log.info(
                    "  [ok] SAM2 fallback ready  (%.1fs)  cache=%s",
                    time.monotonic() - t0,
                    local_dir,
                )
                return
            except Exception as exc2:
                log.warning("  SAM2 fallback also failed: %s", exc2)
                raise exc2 from exc
        raise
