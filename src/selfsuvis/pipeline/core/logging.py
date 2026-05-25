import atexit
import datetime
import logging
import logging.handlers
import os
import queue
import sys

from .log_analytics import LogAnalyticsFilter

_CONFIGURED = False
_ANALYTICS_FILTER = LogAnalyticsFilter()
_queue_listener: "logging.handlers.QueueListener | None" = None
_queue_stopped = False
_sink: "logging.StreamHandler | None" = None


class _CompactFormatter(logging.Formatter):
    """mm:ss,mmm LEVEL leaf_name   message  (date printed once at startup)."""

    def formatTime(self, record: logging.LogRecord, datefmt: "str | None" = None) -> str:
        ct = datetime.datetime.fromtimestamp(record.created)
        return f"{ct.minute:02d}:{ct.second:02d},{ct.microsecond // 1000:03d}"

    def format(self, record: logging.LogRecord) -> str:
        # Avoid mutating the original record (it may be reused by other handlers).
        copy = logging.makeLogRecord(record.__dict__)
        copy.name = record.name.split(".")[-1]
        return super().format(copy)


def configure_logging() -> None:
    """Set up a QueueHandler so concurrent pipeline threads never interleave log lines."""
    global _CONFIGURED, _queue_listener
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Drain any handlers already attached by third-party imports before we
    # install our own, so basicConfig's idempotency guard doesn't block us.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = _CompactFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    # Real sink: one StreamHandler on stderr (avoids mixing with YOLO/subprocess
    # stdout writes that cause line interleaving on TTYs).
    global _sink
    sink = logging.StreamHandler(sys.stderr)
    _sink = sink
    sink.setFormatter(fmt)

    # QueueHandler serialises records from all threads through a single consumer.
    log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    queue_handler = logging.handlers.QueueHandler(log_queue)  # type: ignore[arg-type]
    _queue_listener = logging.handlers.QueueListener(log_queue, sink, respect_handler_level=True)
    _queue_listener.start()
    atexit.register(_stop_listener)

    # Analytics filter runs synchronously on the QueueHandler (calling thread)
    # so snapshot() sees counts immediately after log calls.
    queue_handler.addFilter(_ANALYTICS_FILTER)

    root.addHandler(queue_handler)
    root.setLevel(level)

    # Print the full date+time once so relative mm:ss timestamps are anchored.
    # Skip when the process is only printing --help (argparse will sys.exit immediately).
    if "--help" not in sys.argv and "-h" not in sys.argv:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sink.stream.write(f"Pipeline started: {now}\n")
        sink.stream.flush()

    _CONFIGURED = True


def _stop_listener() -> None:
    global _queue_stopped
    if _queue_listener is not None and not _queue_stopped:
        _queue_listener.stop()
        _queue_stopped = True


def log_pipeline_finished(elapsed_sec: float) -> None:
    """Drain the async log queue, then write the pipeline-finished footer to the sink."""
    global _queue_stopped
    configure_logging()
    _stop_listener()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    delta = str(datetime.timedelta(seconds=int(elapsed_sec)))
    if _sink is not None:
        _sink.stream.write(f"Pipeline finished {now}, overall run time {delta}\n")
        _sink.stream.flush()


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
