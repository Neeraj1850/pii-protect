"""
Live Groq API tests: mask PII with pii-protect, send only the masked text
to Groq (llama-3.1-8b-instant) via LangChain, and verify raw PII never
appears in the prompt sent or the response received.

Requires GROQ_API_KEY (see tests/llm/live/conftest.py for the skip
behaviour when it's absent). Uses the synthetic_rows/kaggle_rows fixtures
from the root tests/conftest.py and the `engine` fixture (InMemoryStorage
+ fresh key) also from the root conftest.
"""
import pytest
from langchain_core.messages import SystemMessage, HumanMessage

from .conftest import skip_no_key

pytestmark = [pytest.mark.llm_integration, pytest.mark.asyncio]


def _build_text(row):
    return (
        f"Email: {row['Email']}. Phone: {row['Phone_Number']}. "
        f"Credit card: {row['Credit_Card']}. PAN: {row['PAN']}. "
        f"Aadhaar: {row['Aadhaar']}. GSTIN: {row['GSTIN']}. "
        f"Mobile: {row['Mobile_IN']}."
    )


@skip_no_key
class TestLLMReceivesMaskedData:
    """Verify that the LLM only ever sees masked (tokenised) PII."""

    async def test_masked_prompt_has_no_raw_email(self, engine, synthetic_rows):
        for row in synthetic_rows[:5]:
            result = await engine.mask(_build_text(row))
            assert row["Email"] not in result.masked_text
            assert "{{" in result.masked_text, "Expected token placeholders"

    async def test_masked_prompt_has_no_raw_pan(self, engine, synthetic_rows):
        for row in synthetic_rows[:5]:
            result = await engine.mask(_build_text(row))
            assert row["PAN"] not in result.masked_text

    async def test_masked_prompt_has_no_raw_credit_card(self, engine, synthetic_rows):
        for row in synthetic_rows[:5]:
            result = await engine.mask(_build_text(row))
            assert row["Credit_Card"] not in result.masked_text

    async def test_masked_kaggle_query_has_no_labeled_pii(self, engine, kaggle_rows):
        """Kaggle ticket queries: all labeled PII must be removed after masking."""
        for row in kaggle_rows[:5]:
            result = await engine.mask(row["customer_query"])
            pii = row["labeled_pii_dict"]
            for key in ["email", "pan", "gstin"]:
                if key in pii:
                    assert pii[key] not in result.masked_text, (
                        f"{row['ticket_id']}: {key} leaked in masked text"
                    )


@skip_no_key
class TestLLMResponseContainsNoPII:
    """Send masked text to Groq and verify original PII doesn't appear in the response."""

    async def test_llm_response_no_email_leak(self, engine, llm, synthetic_rows):
        row = synthetic_rows[0]
        masked_result = await engine.mask(_build_text(row))

        messages = [
            SystemMessage(content="You are a helpful assistant. Summarize the customer information."),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)

        assert row["Email"] not in response.content
        assert row["Credit_Card"] not in response.content
        assert row["PAN"] not in response.content

    async def test_llm_response_no_kaggle_pii_leak(self, engine, llm, kaggle_rows):
        row = kaggle_rows[0]
        masked_result = await engine.mask(row["customer_query"])

        messages = [
            SystemMessage(content="You are a customer support agent. Respond to this ticket."),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)
        response_text = response.content

        pii = row["labeled_pii_dict"]
        for key, value in pii.items():
            if key in ("name", "city", "age"):
                continue  # LLM might infer common names/cities
            assert value not in response_text, f"LLM response leaked {key}='{value}'"

    async def test_llm_unmask_recovers_after_response(self, engine, llm, synthetic_rows):
        """Full pipeline: mask -> LLM -> unmask recovers original values in the response."""
        row = synthetic_rows[1]
        masked_result = await engine.mask(_build_text(row))

        messages = [
            SystemMessage(content=(
                "Repeat the customer details you were given exactly as provided. "
                "Do not change any values."
            )),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)
        unmasked_response = await engine.unmask(response.content)

        if "{{" in response.content:
            assert row["Email"] in unmasked_response or row["PAN"] in unmasked_response


@skip_no_key
class TestLLMBatchPIIShield:
    """Batch processing: mask multiple documents, send to the LLM, verify safety."""

    async def test_batch_mask_and_query(self, engine, llm, synthetic_rows):
        batch = synthetic_rows[:5]
        original_emails = [row["Email"] for row in batch]
        original_pans = [row["PAN"] for row in batch]

        masked_texts = []
        for row in batch:
            result = await engine.mask(_build_text(row))
            masked_texts.append(result.masked_text)

        messages = [
            SystemMessage(content="Summarize each customer record briefly."),
            HumanMessage(content="\n---\n".join(masked_texts)),
        ]
        response = llm.invoke(messages)

        for email in original_emails:
            assert email not in response.content, f"Batch response leaked email: {email}"
        for pan in original_pans:
            assert pan not in response.content, f"Batch response leaked PAN: {pan}"
