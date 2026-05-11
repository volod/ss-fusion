"""Optional dependency helpers.

These helpers keep modules importable in minimal environments by deferring
optional imports until the code path is actually used.
"""


def require_cv2():
    """Return the cv2 module or raise a clear error."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV (cv2) is required for this operation. Install with: pip install opencv-python"
        ) from exc
    return cv2


def require_qdrant_models():
    """Return qdrant_client.http.models or raise a clear error."""
    try:
        from qdrant_client.http import models as qmodels  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Qdrant client is required for this operation. Install with: pip install qdrant-client"
        ) from exc
    return qmodels


def require_qdrant_client():
    """Return QdrantClient class or raise a clear error."""
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Qdrant client is required for this operation. Install with: pip install qdrant-client"
        ) from exc
    return QdrantClient

