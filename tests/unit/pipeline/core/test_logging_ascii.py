import logging


def test_compact_formatter_sanitizes_non_ascii_message():
    from selfsuvis.pipeline.core.logging import _CompactFormatter

    formatter = _CompactFormatter("%(message)s")
    record = logging.LogRecord(
        name="pipeline.local",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Step -> %s",
        args=("DINOv3 -> EfficientViT x2 [ok]",),
        exc_info=None,
    )
    record.msg = "Step → %s"
    record.args = ("DINOv3 → EfficientViT ×2 ✅",)

    assert formatter.format(record) == "Step -> DINOv3 -> EfficientViT x2 [ok]"


