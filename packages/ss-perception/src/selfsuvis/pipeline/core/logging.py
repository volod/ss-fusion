"""Pipeline logging: compact ASCII lines on stderr, with in-process analytics.

Importing this module configures nothing. `configure_logging()` installs the shared
`ss_kit.logging` queue handler plus the pipeline analytics filter.
"""

import datetime
import logging
import sys

from ss_kit.logging import (
    CompactFormatter,
    ascii_log_text,
    stop_logging,
    write_line,
)
from ss_kit.logging import (
    configure_logging as configure_kit_logging,
)
from ss_kit.logging import (
    get_logger as kit_get_logger,
)

from .log_analytics import LogAnalyticsFilter

_CompactFormatter = CompactFormatter
_ascii_log_text = ascii_log_text
_ANALYTICS_FILTER = LogAnalyticsFilter()
_CONFIGURED = False


def configure_logging() -> None:
    """Set up a QueueHandler so concurrent pipeline threads never interleave log lines."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    banner = None if "--help" in sys.argv or "-h" in sys.argv else "Pipeline started"
    configure_kit_logging(filters=[_ANALYTICS_FILTER], banner=banner, force=True)
    _CONFIGURED = True


def log_pipeline_finished(elapsed_sec: float) -> None:
    """Drain the async log queue, then write the pipeline-finished footer to the sink."""
    configure_logging()
    stop_logging()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    delta = str(datetime.timedelta(seconds=int(elapsed_sec)))
    write_line(f"Pipeline finished {now}, overall run time {delta}")


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return kit_get_logger(name)
