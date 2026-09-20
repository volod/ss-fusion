"""Shared probability utilities for sensor fusion."""


def probability_union(scores: list[float]) -> float:
    """Compute the probability that at least one of the events occurred.

    Formula: 1 - product(1 - p for p in scores)
    Returns 0.0 for an empty list; 1.0 if any score is 1.0.
    """
    if not scores:
        return 0.0
    remaining = 1.0
    for p in scores:
        remaining *= 1.0 - p
    return 1.0 - remaining
