"""PII detection tests for texts with mixed entity types."""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.detection, pytest.mark.asyncio]


class TestMixedPII:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_full_business_document(self, engine, sample_text_with_mixed_pii):
        result = await engine.mask(sample_text_with_mixed_pii)
        assert "billing@acme.com" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text
        assert "ABCDE1234F" not in result.masked_text
        assert "+91 98765 43210" not in result.masked_text
        assert result.token_count >= 4

    async def test_pii_at_text_start(self, engine):
        result = await engine.mask("john@acme.com is the contact person.")
        assert "john@acme.com" not in result.masked_text

    async def test_pii_at_text_end(self, engine):
        result = await engine.mask("Contact person: john@acme.com")
        assert "john@acme.com" not in result.masked_text

    async def test_all_pii_sentence(self, engine):
        text = "john@acme.com 27AAPFU0939F1ZV ABCDE1234F +91 98765 43210"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text
        assert "ABCDE1234F" not in result.masked_text

    async def test_pii_in_csv_format(self, engine):
        text = "john@acme.com,27AAPFU0939F1ZV,ABCDE1234F,+91 98765 43210"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text

    async def test_pii_in_json_like_string(self, engine):
        text = '{"email":"john@acme.com","gst":"27AAPFU0939F1ZV"}'
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text

    async def test_pii_in_log_format(self, engine):
        text = "2026-07-06 10:30:00 INFO User john@acme.com logged in from GST 27AAPFU0939F1ZV"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text

    async def test_roundtrip_full_document(self, engine, sample_text_with_mixed_pii):
        result = await engine.mask(sample_text_with_mixed_pii)
        restored = await engine.unmask(result.masked_text)
        assert restored == sample_text_with_mixed_pii
