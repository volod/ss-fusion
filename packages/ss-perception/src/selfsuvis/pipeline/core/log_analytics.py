"""In-process log analytics shared across the codebase."""

import logging
from collections import Counter
from threading import Lock


class LogAnalyticsCollector:
    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str]] = Counter()
        self._lock = Lock()

    def record(self, logger_name: str, level_name: str) -> None:
        with self._lock:
            self._counts[(logger_name, level_name)] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            result: dict[str, dict[str, int]] = {}
            for (logger_name, level_name), count in self._counts.items():
                result.setdefault(logger_name, {})[level_name] = count
            return result

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


_collector = LogAnalyticsCollector()


class LogAnalyticsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        _collector.record(record.name, record.levelname)
        return True


def get_log_analytics() -> LogAnalyticsCollector:
    return _collector
