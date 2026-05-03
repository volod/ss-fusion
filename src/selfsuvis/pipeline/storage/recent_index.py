import time

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class RecentEmbeddingIndex:
    def __init__(self, dim: int, max_size: int, ttl_sec: float):
        self.dim = dim
        self.max_size = max_size
        self.ttl_sec = ttl_sec
        self.vectors: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.index = faiss.IndexFlatIP(dim) if faiss else None

    def _prune(self, now: float) -> None:
        if not self.vectors:
            return
        keep = []
        keep_ts = []
        for v, ts in zip(self.vectors, self.timestamps):
            if now - ts <= self.ttl_sec:
                keep.append(v)
                keep_ts.append(ts)
        self.vectors = keep[-self.max_size :]
        self.timestamps = keep_ts[-self.max_size :]
        if self.index is not None:
            self.index.reset()
            if self.vectors:
                mat = np.vstack(self.vectors).astype(np.float32)
                self.index.add(mat)

    def add(self, vecs: np.ndarray) -> None:
        now = time.time()
        for v in vecs:
            self.vectors.append(v.astype(np.float32))
            self.timestamps.append(now)
        self._prune(now)
        if self.index is not None and vecs.size:
            self.index.add(vecs.astype(np.float32))

    def max_cosine(self, vec: np.ndarray) -> float:
        now = time.time()
        self._prune(now)
        if not self.vectors:
            return -1.0
        if self.index is not None:
            sims, _ = self.index.search(vec.reshape(1, -1).astype(np.float32), 1)
            return float(sims[0][0])
        mat = np.vstack(self.vectors)
        sims = mat @ vec.reshape(-1, 1)
        return float(sims.max())
