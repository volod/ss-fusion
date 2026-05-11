import numpy as np
from typing import TYPE_CHECKING

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.core.optional_deps import require_qdrant_client, require_qdrant_models

if TYPE_CHECKING:
    from qdrant_client.http import models as qmodels  # pragma: no cover


class QdrantStore:
    def __init__(self, clip_dim: int, dino_dim: int | None = None):
        self.logger = get_logger(__name__)
        QdrantClient = require_qdrant_client()
        self._qmodels = require_qdrant_models()
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            check_compatibility=False,
        )
        self.collection = settings.QDRANT_COLLECTION
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        qmodels = self._qmodels
        try:
            from qdrant_client.http.exceptions import UnexpectedResponse  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Qdrant client is required for this operation. Install with: pip install qdrant-client"
            ) from exc

        vectors_config = {
            "clip": qmodels.VectorParams(size=self.clip_dim, distance=qmodels.Distance.COSINE)
        }
        if self.dino_dim:
            vectors_config["dino"] = qmodels.VectorParams(
                size=self.dino_dim, distance=qmodels.Distance.COSINE
            )
        exists = False
        try:
            exists = self.client.collection_exists(self.collection)
        except UnexpectedResponse:
            try:
                self.client.get_collection(self.collection)
                exists = True
            except (UnexpectedResponse, Exception):
                exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config,
            )
            self.logger.info("Created Qdrant collection: %s", self.collection)
        else:
            self.logger.info("Using existing Qdrant collection: %s", self.collection)

    def upsert_points(self, points: list["qmodels.PointStruct"]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector_name: str,
        query_vector: np.ndarray,
        limit: int,
        payload_filter: "qmodels.Filter | None" = None,
    ) -> list["qmodels.ScoredPoint"]:
        qmodels = self._qmodels
        # qdrant-client >= 1.7 removed client.search(); use query_points() instead.
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            using=vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=payload_filter,
        )
        return response.points
