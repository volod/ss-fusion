"""In-memory cosine-similarity nearest-neighbour store.

Drop-in fallback for QdrantStore when Qdrant is not running.
Used by the local full-analysis pipeline and any offline/test context.
"""
from typing import Any, Dict, List

import numpy as np


class InMemoryStore:
    """Cosine-similarity nearest-neighbour store backed by numpy arrays."""

    def __init__(self) -> None:
        self._embeddings: List[np.ndarray] = []
        self._payloads: List[Dict[str, Any]] = []

    def add(self, embedding: np.ndarray, payload: Dict[str, Any]) -> None:
        norm = np.linalg.norm(embedding)
        self._embeddings.append(embedding / norm if norm > 0 else embedding)
        self._payloads.append(payload)

    def search(self, query: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []
        matrix = np.stack(self._embeddings)        # (N, D)
        q = query / (np.linalg.norm(query) + 1e-9)
        scores = matrix @ q                         # (N,)
        top = np.argsort(scores)[::-1][:limit]
        return [
            {"score": float(scores[i]), "payload": self._payloads[i]}
            for i in top
        ]

    def __len__(self) -> int:
        return len(self._embeddings)
