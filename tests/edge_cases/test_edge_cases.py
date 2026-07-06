"""Edge case tests for pii-protect.

Tests cover:
- Empty and whitespace-only inputs
- Very long texts
- Unicode and multilingual text
- Special characters and escaping
- Repeated masking of same text
- Text with only PII (no surrounding text)
- Newlines, tabs, and control characters
- Very short text
"""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.edge_case, pytest.mark.asyncio]


class TestEmptyAndWhitespace:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_empty_string_mask(self, engine):
        result = await engine.mask("")
        assert result.masked_text == ""
        assert result.token_count == 0

    async def test_empty_string_unmask(self, engine):
        result = await engine.unmask("")
        assert result == ""

    def test_empty_string_redact(self, engine):
        assert engine.redact("") == ""

    async def test_whitespace_only(self, engine):
        for ws in ["   ", "\t\t", "\n\n", "  \n\t  "]:
            result = await engine.mask(ws)
            assert result.masked_text == ws
            assert result.token_count == 0

    async def test_single_character(self, engine):
        result = await engine.mask("a")
        assert result.masked_text == "a"
        assert result.token_count == 0


class TestUnicodeAndMultilingual:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_unicode_surrounding_pii(self, engine):
        """PII within unicode text should still be detected."""
        text = "こんにちは john@acme.com お元気ですか"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "こんにちは" in result.masked_text

    async def test_hindi_text_with_pii(self, engine):
        text = "कृपया john@acme.com पर संपर्क करें, GST: 27AAPFU0939F1ZV"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text
        assert "कृपया" in result.masked_text

    async def test_emoji_with_pii(self, engine):
        text = "📧 john@acme.com 📞 +91 98765 43210"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "📧" in result.masked_text

    async def test_unicode_roundtrip(self, engine):
        original = "联系 john@acme.com 关于 GST 27AAPFU0939F1ZV 的事宜"
        result = await engine.mask(original)
        restored = await engine.unmask(result.masked_text)
        assert restored == original


class TestSpecialCharacters:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_html_tags_with_pii(self, engine):
        text = "<p>Contact <a href='mailto:john@acme.com'>john@acme.com</a></p>"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text

    async def test_markdown_with_pii(self, engine):
        text = "**Email**: john@acme.com\n**GST**: 27AAPFU0939F1ZV"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text

    async def test_json_string_with_pii(self, engine):
        import json
        data = {"email": "john@acme.com", "gst": "27AAPFU0939F1ZV"}
        text = json.dumps(data)
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text

    async def test_url_encoded_pii(self, engine):
        text = "User email is john%40acme.com"
        result = await engine.mask(text)
        # URL-encoded @ may or may not be detected - this tests graceful handling
        assert result is not None

    async def test_sql_like_text_with_pii(self, engine):
        text = "INSERT INTO users (email) VALUES ('john@acme.com');"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text


class TestLargeTexts:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_very_large_text_no_pii(self, engine):
        """Large text with no PII should pass through efficiently."""
        text = "Lorem ipsum dolor sit amet. " * 10_000
        result = await engine.mask(text)
        assert result.token_count == 0
        assert len(result.masked_text) == len(text)

    async def test_large_text_with_scattered_pii(self, engine):
        """PII scattered throughout a large document."""
        chunks = []
        for i in range(100):
            chunks.append(f"Section {i}: filler text " * 10)
            if i % 10 == 0:
                chunks.append(f"Contact person{i}@company.com for details.")
        text = "\n".join(chunks)
        result = await engine.mask(text)
        assert result.token_count >= 10
        assert "person0@company.com" not in result.masked_text

    async def test_many_unique_pii_values(self, engine):
        """Text with many different PII values."""
        lines = [f"User {i}: user{i}@test.com" for i in range(100)]
        text = "\n".join(lines)
        result = await engine.mask(text)
        assert result.token_count >= 100
        for i in range(100):
            assert f"user{i}@test.com" not in result.masked_text


class TestRepeatedOperations:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_mask_same_text_repeatedly(self, engine):
        """Masking the same text multiple times should be consistent."""
        text = "john@acme.com"
        results = [await engine.mask(text) for _ in range(10)]
        tokens = [r.masked_text for r in results]
        assert len(set(tokens)) == 1, "Same text should produce same masked output"

    async def test_mask_then_unmask_repeatedly(self, engine):
        original = "Contact john@acme.com about GST 27AAPFU0939F1ZV"
        for _ in range(5):
            result = await engine.mask(original)
            restored = await engine.unmask(result.masked_text)
            assert restored == original

    async def test_redact_same_text_repeatedly(self, engine):
        text = "john@acme.com and 27AAPFU0939F1ZV"
        results = [engine.redact(text) for _ in range(10)]
        assert len(set(results)) == 1, "Redaction should be deterministic"
