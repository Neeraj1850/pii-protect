"""LLM-specific tests: unmasking PII in LLM responses.

Tests cover:
- Unmask tokens that LLM echoes back in its response
- LLM response with mixed tokens and natural text
- Partial token presence
- HTML/markdown formatted responses with tokens
"""
import re
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.llm, pytest.mark.asyncio]


class TestLLMResponseUnmask:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_unmask_echoed_tokens(self, engine):
        original = "Contact john@acme.com about GST 27AAPFU0939F1ZV"
        mask_result = await engine.mask(original)
        email_token = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        gst_token = re.findall(r"\{\{GST:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        llm_response = f"I will contact {email_token} regarding their GST {gst_token}."
        unmasked = await engine.unmask(llm_response)
        assert "john@acme.com" in unmasked
        assert "27AAPFU0939F1ZV" in unmasked

    async def test_unmask_mixed_response(self, engine):
        mask_result = await engine.mask("Email: john@acme.com")
        email_token = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        llm_response = (
            f"Based on my analysis, the email address {email_token} belongs to "
            f"a customer in the ACME corporation. I recommend sending a follow-up "
            f"to {email_token} within 24 hours."
        )
        unmasked = await engine.unmask(llm_response)
        assert unmasked.count("john@acme.com") == 2

    async def test_unmask_markdown_response(self, engine):
        mask_result = await engine.mask("Customer: john@acme.com, PAN: ABCDE1234F")
        email_token = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        pan_token = re.findall(r"\{\{PAN:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        llm_response = (
            f"## Customer Report\n\n"
            f"- **Email**: {email_token}\n"
            f"- **PAN**: {pan_token}\n\n"
            f"Please verify the details above."
        )
        unmasked = await engine.unmask(llm_response)
        assert "john@acme.com" in unmasked
        assert "ABCDE1234F" in unmasked
        assert "## Customer Report" in unmasked

    async def test_unmask_response_no_tokens(self, engine):
        response = "Thank you for your inquiry. We will process it soon."
        result = await engine.unmask(response)
        assert result == response

    async def test_unmask_preserves_non_token_text(self, engine):
        mask_result = await engine.mask("john@acme.com")
        email_token = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        prefix = "Dear customer, regarding "
        suffix = ", we have updates."
        llm_response = f"{prefix}{email_token}{suffix}"
        unmasked = await engine.unmask(llm_response)
        assert unmasked.startswith(prefix)
        assert unmasked.endswith(suffix)
        assert "john@acme.com" in unmasked
