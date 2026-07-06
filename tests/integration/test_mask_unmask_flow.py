"""Integration tests for mask() -> unmask() roundtrip flow.

Tests cover:
- Full mask/unmask roundtrip restores original text
- Masked text contains {{TYPE:xxxxx}} tokens, no raw PII
- MaskResult fields: masked_text, token_count, entity_counts, entities
- Multiple PII types in one text
- Repeated PII values produce same token (deduplication)
- Unmask with no matching tokens returns text with [UNRESOLVED]
- Concurrent mask operations
"""
import re
import asyncio
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestMaskUnmaskRoundtrip:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_basic_roundtrip(self, engine):
        original = "Contact john@acme.com about GST 27AAPFU0939F1ZV"
        result = await engine.mask(original)
        assert "john@acme.com" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text
        restored = await engine.unmask(result.masked_text)
        assert restored == original

    async def test_mask_result_has_token_count(self, engine):
        result = await engine.mask("Email john@acme.com and GST 27AAPFU0939F1ZV")
        assert result.token_count >= 2

    async def test_mask_result_has_entity_counts(self, engine):
        result = await engine.mask("Email john@acme.com and GST 27AAPFU0939F1ZV")
        assert isinstance(result.entity_counts, dict)
        assert "EMAIL" in result.entity_counts
        assert "GST" in result.entity_counts

    async def test_mask_result_has_entities(self, engine):
        result = await engine.mask("Contact john@acme.com")
        assert hasattr(result, 'entities')
        assert len(result.entities) >= 1

    async def test_masked_text_format(self, engine):
        result = await engine.mask("john@acme.com")
        tokens = re.findall(r"\{\{[A-Z_]+:[a-f0-9]+\}\}", result.masked_text)
        assert len(tokens) >= 1

    async def test_repeated_value_same_token(self, engine):
        text = "CC john@acme.com and BCC john@acme.com"
        result = await engine.mask(text)
        tokens = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", result.masked_text)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1], "Same value should produce same token"

    async def test_roundtrip_with_multiple_types(self, engine):
        original = (
            "Invoice for john@acme.com, GST 27AAPFU0939F1ZV, "
            "PAN ABCDE1234F, call +91 98765 43210"
        )
        result = await engine.mask(original)
        assert result.token_count >= 4
        restored = await engine.unmask(result.masked_text)
        assert restored == original

    async def test_no_pii_passthrough(self, engine):
        text = "The weather is lovely today."
        result = await engine.mask(text)
        assert result.masked_text == text
        assert result.token_count == 0

    async def test_unmask_unknown_tokens_unresolved(self, engine):
        fake_masked = "Hello {{EMAIL:fffff}}"
        result = await engine.unmask(fake_masked)
        assert "UNRESOLVED" in result or "{{EMAIL:fffff}}" in result

    async def test_concurrent_mask_operations(self, engine):
        texts = [f"Contact user{i}@example.com" for i in range(10)]
        results = await asyncio.gather(*[engine.mask(t) for t in texts])
        for i, result in enumerate(results):
            assert f"user{i}@example.com" not in result.masked_text
            restored = await engine.unmask(result.masked_text)
            assert restored == texts[i]

    async def test_roundtrip_preserves_whitespace(self, engine):
        original = "  john@acme.com  \n  +91 98765 43210  \t  "
        result = await engine.mask(original)
        restored = await engine.unmask(result.masked_text)
        assert restored == original
