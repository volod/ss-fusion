from selfsuvis.pipeline.core.log_analytics import get_log_analytics
from selfsuvis.pipeline.core.logging import get_logger


def test_log_analytics_collects_counts():
    collector = get_log_analytics()
    collector.reset()

    logger = get_logger("test.analytics")
    logger.setLevel("INFO")
    logger.info("one")
    logger.warning("two")

    snapshot = collector.snapshot()

    assert snapshot["test.analytics"]["INFO"] >= 1
    assert snapshot["test.analytics"]["WARNING"] >= 1
