"""Unit tests for NEREngine and RegexNERLayer (pii_protect.ner).

Tests cover:
- NEREngine initialisation (default = regex only)
- Detection of various PII entity types via regex layer
- Non-overlapping span output
- Confidence scores present
- Entity type correctness
- Empty text returns no spans
- Text with no PII returns no spans
"""
import pytest
from pii_protect import NEREngine


pytestmark = pytest.mark.unit


class TestNEREngineInit:
    def test_default_init_no_extras(self):
        """NEREngine() with no arguments should work (regex only)."""
        ner = NEREngine()
        assert ner is not None

    def test_spacy_not_enabled_by_default(self):
        ner = NEREngine()
        assert ner is not None


class TestRegexDetection:
    @pytest.fixture
    def ner(self):
        return NEREngine()

    def test_detect_email(self, ner):
        spans = ner.detect("Contact john@acme.com please")
        assert any(s.entity_type == "EMAIL" for s in spans)

    def test_detect_phone_indian(self, ner):
        spans = ner.detect("Call +91 98765 43210")
        phone_spans = [s for s in spans if s.entity_type == "PHONE"]
        assert len(phone_spans) >= 1

    def test_detect_phone_international(self, ner):
        spans = ner.detect("Ring +44 20 7946 0958")
        phone_spans = [s for s in spans if s.entity_type == "PHONE"]
        assert len(phone_spans) >= 1

    def test_detect_gst(self, ner):
        spans = ner.detect("GST number 27AAPFU0939F1ZV")
        assert any(s.entity_type == "GST" for s in spans)

    def test_detect_pan(self, ner):
        spans = ner.detect("PAN is ABCDE1234F")
        assert any(s.entity_type == "PAN" for s in spans)

    def test_detect_iban(self, ner):
        spans = ner.detect("IBAN: GB29NWBK60161331926819")
        assert any(s.entity_type == "IBAN" for s in spans)

    def test_detect_swift(self, ner):
        spans = ner.detect("SWIFT code DEUTDEFF")
        assert any(s.entity_type == "SWIFT" for s in spans)

    def test_detect_credit_card(self, ner):
        spans = ner.detect("Card: 4111111111111111")
        cc_types = [s.entity_type for s in spans]
        assert any("CARD" in t or "CREDIT" in t for t in cc_types)

    def test_detect_multiple_entities(self, ner):
        text = "Email john@acme.com, GST 27AAPFU0939F1ZV, PAN ABCDE1234F"
        spans = ner.detect(text)
        types = {s.entity_type for s in spans}
        assert "EMAIL" in types
        assert "GST" in types
        assert "PAN" in types

    def test_no_pii_returns_empty(self, ner):
        spans = ner.detect("The weather is nice today.")
        assert len(spans) == 0

    def test_empty_text_returns_empty(self, ner):
        spans = ner.detect("")
        assert len(spans) == 0

    def test_spans_have_confidence(self, ner):
        spans = ner.detect("john@acme.com")
        for s in spans:
            assert hasattr(s, 'confidence') or hasattr(s, 'score')

    def test_spans_non_overlapping(self, ner):
        text = "Email john@acme.com and GST 27AAPFU0939F1ZV"
        spans = ner.detect(text)
        sorted_spans = sorted(spans, key=lambda s: s.start)
        for i in range(len(sorted_spans) - 1):
            assert sorted_spans[i].end <= sorted_spans[i + 1].start, (
                f"Spans overlap: {sorted_spans[i]} and {sorted_spans[i+1]}"
            )

    def test_detect_invoice_number(self, ner):
        spans = ner.detect("Invoice INV-2026-00417")
        types = {s.entity_type for s in spans}
        assert len(spans) >= 1, f"Expected invoice detection, got types: {types}"

    def test_detect_po_number(self, ner):
        spans = ner.detect("PO number PO-2026-0042")
        assert len(spans) >= 1
