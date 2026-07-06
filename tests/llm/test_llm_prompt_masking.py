"""LLM-specific tests: masking PII before sending prompts to an LLM.

Tests cover:
- System prompt remains unmasked (no PII in instructions)
- User prompt PII is fully masked
- Masked prompt is safe to send to external LLM
- Roundtrip: mask user input -> (simulate LLM) -> unmask response
- Multi-turn conversation masking
- Structured prompt templates with PII
- Batch document masking for RAG pipelines
"""
import re
import asyncio
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.llm, pytest.mark.asyncio]


class TestLLMPromptMasking:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_system_prompt_no_pii_unchanged(self, engine):
        system_prompt = (
            "You are a helpful customer service assistant. "
            "Analyse the complaint and suggest a resolution."
        )
        result = await engine.mask(system_prompt)
        assert result.masked_text == system_prompt
        assert result.token_count == 0

    async def test_user_prompt_pii_masked(self, engine, sample_llm_prompt):
        result = await engine.mask(sample_llm_prompt)
        assert "priya.sharma@techcorp.in" not in result.masked_text
        assert "29GGGGG1314R9Z6" not in result.masked_text
        assert "BSRPS5432K" not in result.masked_text
        assert result.token_count >= 3

    async def test_masked_prompt_safe_for_llm(self, engine):
        text = (
            "Customer John Smith (john@smith.com, PAN: ABCDE1234F) "
            "called about invoice INV-2026-789 worth 10L."
        )
        pii_values = ["john@smith.com", "ABCDE1234F"]
        result = await engine.mask(text)
        for val in pii_values:
            assert val not in result.masked_text, f"PII '{val}' leaked into masked text"

    async def test_llm_roundtrip_simulation(self, engine):
        user_input = "Summarise: john@acme.com filed complaint about GST 27AAPFU0939F1ZV"
        mask_result = await engine.mask(user_input)
        email_token = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        gst_token = re.findall(r"\{\{GST:[a-f0-9]+\}\}", mask_result.masked_text)[0]
        llm_response = f"Summary: The complaint from {email_token} concerns GST number {gst_token}."
        unmasked_response = await engine.unmask(llm_response)
        assert "john@acme.com" in unmasked_response
        assert "27AAPFU0939F1ZV" in unmasked_response

    async def test_multi_turn_conversation(self, engine):
        scope = "conversation-001"
        t1 = await engine.mask(
            "My email is john@acme.com and my GST is 27AAPFU0939F1ZV", scope=scope,
        )
        t2 = await engine.mask("Also forward to john@acme.com please", scope=scope)
        email_tokens_t1 = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", t1.masked_text)
        email_tokens_t2 = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", t2.masked_text)
        assert email_tokens_t1[0] == email_tokens_t2[0]

    async def test_rag_batch_document_masking(self, engine):
        documents = [
            "Invoice from john@acme.com for GST 27AAPFU0939F1ZV",
            "Receipt: paid to jane@corp.com, PAN ZZZZZ9999Z",
            "Contract with Acme Corp, contact ceo@acme.com",
        ]
        pii_per_doc = [
            ["john@acme.com", "27AAPFU0939F1ZV"],
            ["jane@corp.com", "ZZZZZ9999Z"],
            ["ceo@acme.com"],
        ]
        for i, doc in enumerate(documents):
            result = await engine.mask(doc)
            for pii_val in pii_per_doc[i]:
                assert pii_val not in result.masked_text

    async def test_structured_prompt_template(self, engine):
        prompt = (
            "### Context\n"
            "Customer: Rajesh Kumar\n"
            "Email: rajesh@company.in\n"
            "Phone: +91 98765 43210\n"
            "GST: 27AAPFU0939F1ZV\n\n"
            "### Task\n"
            "Generate a follow-up email for the above customer."
        )
        result = await engine.mask(prompt)
        assert "rajesh@company.in" not in result.masked_text
        assert "+91 98765 43210" not in result.masked_text
        assert "27AAPFU0939F1ZV" not in result.masked_text
        assert "### Context" in result.masked_text
        assert "### Task" in result.masked_text

    async def test_json_payload_masking_for_api(self, engine):
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Help customer john@acme.com with GST 27AAPFU0939F1ZV"},
            ],
        }
        masked_payload = await engine.mask_dict(payload)
        user_msg = masked_payload["messages"][1]["content"]
        assert "john@acme.com" not in user_msg
        assert "27AAPFU0939F1ZV" not in user_msg
        assert masked_payload["messages"][0]["content"] == payload["messages"][0]["content"]
