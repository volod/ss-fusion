"""Helpers for suppressing noisy third-party warnings and console output."""

import contextlib
import io
import logging
import warnings
from collections.abc import Iterator, Mapping


@contextlib.contextmanager
def suppress_runtime_noise(
    *patterns: str,
    logger_levels: Mapping[str, int] | None = None,
) -> Iterator[None]:
    """Temporarily silence known warning text and stray stdout/stderr writes.

    Some third-party libraries emit operational warnings directly to stderr or as
    ``warnings.warn(...)`` even when their loggers are configured correctly.
    This keeps local pipeline logs readable without muting our own progress logs.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    previous_levels: dict[str, int] = {}
    transformers_logging = None
    transformers_verbosity = None
    hf_logging = None
    with warnings.catch_warnings():
        for pattern in patterns:
            warnings.filterwarnings("ignore", message=pattern)
        if logger_levels:
            for name, level in logger_levels.items():
                logger = logging.getLogger(name)
                previous_levels[name] = logger.level
                logger.setLevel(level)
        try:
            try:
                from transformers.utils import logging as transformers_logging  # type: ignore

                transformers_verbosity = transformers_logging.get_verbosity()
                transformers_logging.set_verbosity_error()
                if hasattr(transformers_logging, "disable_progress_bar"):
                    transformers_logging.disable_progress_bar()
            except Exception:
                transformers_logging = None
            try:
                from huggingface_hub.utils import logging as hf_logging  # type: ignore

                if hasattr(hf_logging, "disable_progress_bar"):
                    hf_logging.disable_progress_bar()
            except Exception:
                hf_logging = None
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                yield
        finally:
            if transformers_logging is not None and transformers_verbosity is not None:
                transformers_logging.set_verbosity(transformers_verbosity)
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)
