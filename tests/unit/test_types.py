"""Unit tests for pii_protect.types (DetectedSpan convenience methods).

Tests cover:
- .length property (end - start)
- .overlaps() / .contains() helpers
"""
import pytest
from pii_protect.types import DetectedSpan, EntityType


pytestmark = pytest.mark.unit


def make_span(start, end, text="x", entity_type=EntityType.OTHER, confidence=0.9):
    return DetectedSpan(
        start=start, end=end, text=text, entity_type=entity_type,
        confidence=confidence, source="test",
    )


class TestDetectedSpanLength:
    def test_length_property(self):
        span = make_span(5, 10)
        assert span.length == 5

    def test_length_zero_width_span(self):
        span = make_span(5, 5)
        assert span.length == 0


class TestDetectedSpanOverlaps:
    def test_overlapping_spans(self):
        a = make_span(0, 10)
        b = make_span(5, 15)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_non_overlapping_spans(self):
        a = make_span(0, 5)
        b = make_span(5, 10)
        assert not a.overlaps(b)

    def test_contains(self):
        outer = make_span(0, 20)
        inner = make_span(5, 10)
        assert outer.contains(inner)
        assert not inner.contains(outer)
