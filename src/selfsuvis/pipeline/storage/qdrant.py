from typing import Dict, Any, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from selfsuvis.pipeline.core import get_logger, settings


class QdrantStore:
    def __init__(self, clip_dim: int, dino_dim: Optional[int] = None):
        self.logger = get_logger(__name__)
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection = settings.QDRANT_COLLECTION
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        vectors_config = {
            "clip": qmodels.VectorParams(size=self.clip_dim, distance=qmodels.Distance.COSINE)
        }
        if self.dino_dim:
            vectors_config["dino"] = qmodels.VectorParams(size=self.dino_dim, distance=qmodels.Distance.COSINE)
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

    def upsert_points(self, points: List[qmodels.PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector_name: str,
        query_vector: np.ndarray,
        limit: int,
        payload_filter: Optional[qmodels.Filter] = None,
    ) -> List[qmodels.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=qmodels.NamedVector(name=vector_name, vector=query_vector.tolist()),
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=payload_filter,
        )
