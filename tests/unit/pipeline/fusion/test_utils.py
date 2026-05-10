"""Unit tests for pipeline.fusion.utils.probability_union."""

import pytest

from selfsuvis.pipeline.fusion.utils import probability_union


def test_empty_list():
    assert probability_union([]) == 0.0


def test_single_value():
    assert probability_union([0.5]) == pytest.approx(0.5)


def test_two_halves():
    # 1 - (1 - 0.5) * (1 - 0.5) = 1 - 0.25 = 0.75
    assert probability_union([0.5, 0.5]) == pytest.approx(0.75)


def test_certain_value():
    assert probability_union([1.0, 0.3]) == pytest.approx(1.0)


def test_all_zero():
    assert probability_union([0.0, 0.0]) == pytest.approx(0.0)


def test_three_values():
    # 1 - 0.5 * 0.4 * 0.3 = 1 - 0.06 = 0.94
    assert probability_union([0.5, 0.6, 0.7]) == pytest.approx(1 - 0.5 * 0.4 * 0.3)
