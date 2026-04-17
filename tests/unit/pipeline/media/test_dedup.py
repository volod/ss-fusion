"""Unit tests for pipeline.dedup."""

import numpy as np
import pytest

from selfsuvis.pipeline.media.dedup import PhashLRU, dhash


def test_dhash_deterministic():
    pytest.importorskip("cv2", reason="cv2 required for dhash")
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    h1 = dhash(img)
    h2 = dhash(img)
    assert h1 == h2


def test_dhash_different_images():
    pytest.importorskip("cv2", reason="cv2 required for dhash")
    # Use images with different gradients (flat 0/255 yield same dhash)
    img1 = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    img2 = np.tile(np.linspace(255, 0, 64, dtype=np.uint8), (64, 1))
    h1 = dhash(img1)
    h2 = dhash(img2)
    assert h1 != h2


def test_phash_lru_near_duplicate():
    lru = PhashLRU(max_size=100, hamming_max=6)
    h = 0x123456789ABCDEF0
    assert not lru.near_duplicate(h)
    lru.add(h)
    assert lru.near_duplicate(h)
    # Same hash is duplicate
    assert lru.near_duplicate(h)


def test_phash_lru_eviction():
    lru = PhashLRU(max_size=3, hamming_max=0)
    lru.add(1)
    lru.add(2)
    lru.add(3)
    assert lru.near_duplicate(1)
    lru.add(4)
    # 1 should be evicted (FIFO)
    assert not lru.near_duplicate(1)
    assert lru.near_duplicate(2)
    assert lru.near_duplicate(3)
    assert lru.near_duplicate(4)


# --- additional tests ---

def test_phash_lru_hamming_within_threshold():
    """near_duplicate is True when Hamming distance <= hamming_max."""
    lru = PhashLRU(max_size=100, hamming_max=4)
    # Build a base hash where top-16 bits match (same bucket) so lookup fires.
    base = 0xABCD_0000_0000_0000
    # Flip 3 bits in the lower 48 bits → distance 3 <= 4 → duplicate
    similar = base ^ 0b111
    lru.add(base)
    assert lru.near_duplicate(similar)


def test_phash_lru_hamming_outside_threshold():
    """near_duplicate is False when Hamming distance > hamming_max."""
    lru = PhashLRU(max_size=100, hamming_max=2)
    base = 0xABCD_0000_0000_0000
    # Flip 5 bits → distance 5 > 2 → not a duplicate
    far = base ^ 0b11111
    lru.add(base)
    assert not lru.near_duplicate(far)


def test_phash_lru_hamming_max_zero_exact_only():
    """hamming_max=0 accepts only exact matches."""
    lru = PhashLRU(max_size=100, hamming_max=0)
    h = 0xDEAD_BEEF_CAFE_1234
    lru.add(h)
    assert lru.near_duplicate(h)
    # Any bit flip is not a duplicate
    assert not lru.near_duplicate(h ^ 1)


def test_phash_lru_different_bucket_not_matched():
    """Hashes in different buckets are never matched, regardless of hamming_max."""
    # Buckets are determined by bits [48:64] (top 16 bits).
    # Two hashes with different top-16 bits are in different buckets.
    lru = PhashLRU(max_size=100, hamming_max=63)  # huge threshold
    h1 = 0x0001_0000_0000_0000  # bucket 0x0001
    h2 = 0x0002_0000_0000_0000  # bucket 0x0002; Hamming distance = 2
    lru.add(h1)
    # h2 has a different bucket key, so it won't be found even with high threshold
    assert not lru.near_duplicate(h2)


def test_phash_lru_empty_returns_false():
    """near_duplicate on an empty LRU always returns False."""
    lru = PhashLRU(max_size=100, hamming_max=10)
    assert not lru.near_duplicate(0xDEAD_BEEF_0000_1234)


def test_phash_lru_duplicate_add():
    """Adding the same hash twice still reports it as a near_duplicate."""
    lru = PhashLRU(max_size=100, hamming_max=0)
    h = 0x1234_5678_9ABC_DEF0
    lru.add(h)
    lru.add(h)
    assert lru.near_duplicate(h)
    # Size should have grown to 2 in the queue
    assert len(lru.queue) == 2
