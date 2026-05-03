from collections import defaultdict, deque

import numpy as np


class PhashLRU:
    def __init__(self, max_size: int, hamming_max: int):
        self.max_size = max_size
        self.hamming_max = hamming_max
        self.queue: deque[int] = deque()
        self.buckets: dict[int, list[int]] = defaultdict(list)

    def _bucket_key(self, h: int) -> int:
        return (h >> 48) & 0xFFFF

    def add(self, h: int) -> None:
        self.queue.append(h)
        self.buckets[self._bucket_key(h)].append(h)
        if len(self.queue) > self.max_size:
            old = self.queue.popleft()
            bucket = self._bucket_key(old)
            lst = self.buckets[bucket]
            try:
                lst.remove(old)
            except ValueError:
                pass
            if not lst:
                self.buckets.pop(bucket, None)

    def near_duplicate(self, h: int) -> bool:
        bucket = self._bucket_key(h)
        candidates = self.buckets.get(bucket, [])
        for c in candidates:
            if _hamming(h, c) <= self.hamming_max:
                return True
        return False


def dhash(img_gray: np.ndarray, hash_size: int = 8) -> int:
    resized = _resize(img_gray, hash_size + 1, hash_size)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    try:
        import cv2

        if hasattr(cv2, "resize") and hasattr(cv2, "INTER_AREA"):
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:
        pass

    from PIL import Image

    return np.asarray(
        Image.fromarray(img).resize((w, h), resample=Image.Resampling.BOX)
    )


def _hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))
