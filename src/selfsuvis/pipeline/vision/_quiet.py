"""Helpers for suppressing noisy third-party warnings and console output."""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from typing import Iterator, Mapping


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
    with warnings.catch_warnings():
        for pattern in patterns:
            warnings.filterwarnings("ignore", message=pattern)
        if logger_levels:
            for name, level in logger_levels.items():
                logger = logging.getLogger(name)
                previous_levels[name] = logger.level
                logger.setLevel(level)
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                yield
        finally:
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)
