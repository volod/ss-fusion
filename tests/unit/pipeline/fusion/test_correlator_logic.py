"""Unit tests for correlator logic extracted to pure functions."""

import pytest

from selfsuvis.pipeline.fusion.correlator import _risk_level
from selfsuvis.pipeline.fusion.utils import probability_union


def test_risk_level_low():
    assert _risk_level(0.2) == "low"
    assert _risk_level(0.39) == "low"


def test_risk_level_medium():
    assert _risk_level(0.4) == "medium"
    assert _risk_level(0.69) == "medium"


def test_risk_level_high():
    assert _risk_level(0.7) == "high"
    assert _risk_level(0.89) == "high"


def test_risk_level_critical():
    assert _risk_level(0.9) == "critical"
    assert _risk_level(1.0) == "critical"


def test_probability_union_used_for_confidence():
    # Rule has modalities [camera, audio], each event confidence 0.8 and 0.6
    scores = [0.8, 0.6]
    confidence = probability_union(scores)
    # 1 - (1-0.8)*(1-0.6) = 1 - 0.2*0.4 = 1 - 0.08 = 0.92
    assert confidence == pytest.approx(0.92)


def test_confidence_below_min_not_incident():
    # If probability_union result < min_confidence, no incident should be created
    scores = [0.3, 0.2]
    confidence = probability_union(scores)
    min_confidence = 0.7
    assert confidence < min_confidence


def test_confidence_above_min_triggers_incident():
    scores = [0.8, 0.7]
    confidence = probability_union(scores)
    min_confidence = 0.5
    assert confidence >= min_confidence
