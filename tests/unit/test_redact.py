"""Unit tests for the redact() method of PIIMaskingEngine.

redact() is synchronous, touches no storage, and is irreversible.
Tests cover:
- Basic redaction replaces PII with [REDACTED:TYPE]
- Multiple PII types in one text
- No PII text passes through unchanged
- Empty text
- Redacted text contains no original PII values
- Redaction format consistency
"""
import re
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage


pytestmark = pytest.mark.unit


class TestRedact:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(storage=InMemoryStorage()) as eng:
            yield eng

    def test_redact_email(self, engine):
        result = engine.redact("Contact john@acme.com")
        assert "john@acme.com" not in result
        assert "[REDACTED:EMAIL]" in result

    def test_redact_phone(self, engine):
        result = engine.redact("Call +91 98765 43210")
        assert "+91 98765 43210" not in result
        assert "[REDACTED:" in result

    def test_redact_gst(self, engine):
        result = engine.redact("GST: 27AAPFU0939F1ZV")
        assert "27AAPFU0939F1ZV" not in result
        assert "[REDACTED:GST]" in result

    def test_redact_pan(self, engine):
        result = engine.redact("PAN: ABCDE1234F")
        assert "ABCDE1234F" not in result
        assert "[REDACTED:PAN]" in result

    def test_redact_multiple_types(self, engine):
        text = "Email john@acme.com, GST 27AAPFU0939F1ZV, PAN ABCDE1234F"
        result = engine.redact(text)
        assert "john@acme.com" not in result
        assert "27AAPFU0939F1ZV" not in result
        assert "ABCDE1234F" not in result
        markers = re.findall(r"\[REDACTED:[A-Z_]+\]", result)
        assert len(markers) >= 3

    def test_redact_no_pii(self, engine):
        text = "The weather is nice today."
        result = engine.redact(text)
        assert result == text

    def test_redact_empty_string(self, engine):
        result = engine.redact("")
        assert result == ""

    def test_redact_format_consistency(self, engine):
        result = engine.redact("john@acme.com and 27AAPFU0939F1ZV")
        redactions = re.findall(r"\[REDACTED:[A-Z_]+\]", result)
        assert len(redactions) >= 2
        for r in redactions:
            assert re.match(r"^\[REDACTED:[A-Z_]+\]$", r)

    def test_redact_is_synchronous(self, engine):
        """redact() should be a sync method, not a coroutine."""
        import asyncio
        result = engine.redact("john@acme.com")
        assert not asyncio.iscoroutine(result)

    def test_redact_mixed_pii_text(self, engine, sample_text_with_mixed_pii):
        result = engine.redact(sample_text_with_mixed_pii)
        assert "billing@acme.com" not in result
        assert "27AAPFU0939F1ZV" not in result
        assert "ABCDE1234F" not in result
