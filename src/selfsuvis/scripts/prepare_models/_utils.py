"""Shared utilities: noise suppression, label formatting, load-report parsing."""

import contextlib
import io
import os
import re
import sys
import warnings
from pathlib import Path

from selfsuvis.pipeline.core.logging import get_logger

log = get_logger("prepare_models")

# Derived once so every submodule can import a single canonical value.
_CACHE_DIR = Path(os.getenv("CACHE_DIR", "./.data/.cache"))

# Tracks resolved model names already warmed up in this session.
_warmed: set = set()

# Key prefixes present in the full SeamlessM4T checkpoint that are absent in the
# SpeechToText variant — their absence is documented as safe to ignore.
_SEAMLESS_EXPECTED_UNEXPECTED = ("text_encoder.", "t2u_model.", "vocoder.")


@contextlib.contextmanager
def _quiet_hf():
    """Suppress repetitive HF transformers + ultralytics noise during model warmup.

    Redirects stdout/stderr to a sink and sets transformers verbosity to ERROR
    for the duration of the block.  Our own log.info() calls are unaffected
    because they go through the root logging handler, not stdout/stderr directly.
    """
    orig_tf_verbosity = None
    try:
        import transformers

        orig_tf_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass
    sink = io.StringIO()
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter")
            warnings.filterwarnings("ignore", message=".*Using a slow image processor")
            warnings.filterwarnings("ignore", message=".*use_fast.*will be the default")
            warnings.filterwarnings("ignore", message=".*VideoMAEFeatureExtractor is deprecated")
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield
    finally:
        if orig_tf_verbosity is not None:
            try:
                import transformers

                transformers.logging.set_verbosity(orig_tf_verbosity)
            except Exception:
                pass


def _label(model_name: str, resolved: str) -> str:
    if model_name != resolved:
        return f"{model_name} → {resolved} (alias)"
    return model_name


@contextlib.contextmanager
def _capture_hf_load_report(label: str):
    """Capture the [transformers] LOAD REPORT printed on model load.

    Replaces the noisy table with a single log line.  Three categories:
      - known-UNEXPECTED (TTS/T2U components absent by design) -> INFO [ok]
      - unknown-UNEXPECTED (weights outside expected absent set) -> WARNING
      - MISSING (weights needed for the task are absent) -> WARNING
    Non-report stdout/stderr is passed through unchanged.
    Newer transformers sends the LOAD REPORT via logger (stderr); capture both.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        yield
    output = stdout_buf.getvalue() + stderr_buf.getvalue()
    if "LOAD REPORT" not in output:
        passthrough = stdout_buf.getvalue()
        if passthrough.strip():
            sys.stdout.write(passthrough)
        passthrough_err = stderr_buf.getvalue()
        if passthrough_err.strip():
            sys.stderr.write(passthrough_err)
        return
    unexpected: list[str] = []
    missing: list[str] = []
    for line in output.splitlines():
        m = re.match(r"^\s*([\w.{}, 0-9]+?)\s*\|\s*(UNEXPECTED|MISSING)\s*\|", line)
        if m:
            key, status = m.group(1).strip(), m.group(2)
            (unexpected if status == "UNEXPECTED" else missing).append(key)
    known = [k for k in unexpected if k.startswith(_SEAMLESS_EXPECTED_UNEXPECTED)]
    unknown = [k for k in unexpected if not k.startswith(_SEAMLESS_EXPECTED_UNEXPECTED)]
    if known:
        log.info(
            "  [ok] %s: %d TTS/T2U weights skipped (expected for speech-to-text task)",
            label,
            len(known),
        )
    if unknown:
        log.warning(
            "  [warn] %s: %d unexpected weights outside known absent set: %s",
            label,
            len(unknown),
            unknown[:3],
        )
    if missing:
        log.warning(
            "  [warn] %s: %d missing weights — ASR output may be degraded: %s",
            label,
            len(missing),
            missing[:3],
        )


def _is_editable_installed(package_name: str) -> bool:
    """Return True if *package_name* is installed as an editable package."""
    try:
        import importlib.metadata

        importlib.metadata.distribution(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _importable_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False
