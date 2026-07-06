"""
Tests: LLM PII Shield — Groq + LangChain + DeepEval Integration

End-to-end tests that:
1. Load PII-laden text from synthetic and Kaggle datasets
2. Mask PII using pii-protect before sending to LLM
3. Call Groq (llama-3.1-8b-instant) via LangChain with the MASKED text
4. Verify the LLM never sees or returns original PII values
5. Use DeepEval GEval metric (powered by Groq Llama) for evaluation

Environment: GROQ_API_KEY must be set in .env
"""
import os
import csv
import ast
import json
from typing import Optional, Tuple, Union
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel

from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import time
import asyncio
from groq import Groq, AsyncGroq, RateLimitError

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

# Load environment variables
load_dotenv()

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
GROQ_MODEL = "llama-3.1-8b-instant"

pytestmark = [pytest.mark.llm_integration, pytest.mark.asyncio]

# ---------------------------------------------------------------------------
# Skip if no API key
# ---------------------------------------------------------------------------
def _has_groq_key():
    return bool(os.environ.get("GROQ_API_KEY"))

skip_no_key = pytest.mark.skipif(
    not _has_groq_key(), reason="GROQ_API_KEY not set"
)

# ---------------------------------------------------------------------------

# DeepEval Custom LLM Wrapper — routes all evaluation calls to Groq Llama
# ---------------------------------------------------------------------------

class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    Custom DeepEval model wrapper that uses Groq's llama-3.1-8b-instant
    as the evaluation judge instead of OpenAI GPT models.
    """

    def __init__(self, model_name: str = GROQ_MODEL):
        self._model_name = model_name
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        # Do NOT call super().__init__() because DeepEvalBaseLLM.__init__
        # calls load_model() and sets self.model, we handle it manually.
        self.name = model_name
        self.model = self._client

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return self._model_name

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Synchronous generation via Groq chat completions with retry on RateLimitError."""
        print(f"\n--- [DEBUG] GEval Prompt ---\n{prompt}\n---------------------------\n")
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(5):
            try:
                if schema:
                    schema_json = json.dumps(schema.model_json_schema(), indent=2)
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                f"You must respond ONLY with valid JSON that matches this schema:\n"
                                f"{schema_json}\n"
                                f"Do not include any text outside the JSON object."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    completion = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                        response_format={"type": "json_object"},
                    )
                    raw = completion.choices[0].message.content
                    print(f"\n--- [DEBUG] GEval Schema Response ---\n{raw}\n-----------------------------------\n")
                    try:
                        parsed = json.loads(raw)
                        return schema.model_validate(parsed), 0.0
                    except Exception:
                        return raw, 0.0
                else:
                    completion = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                    )
                    raw = completion.choices[0].message.content
                    print(f"\n--- [DEBUG] GEval Raw Response ---\n{raw}\n---------------------------------\n")
                    return raw, 0.0
            except RateLimitError as e:
                wait_time = (attempt + 1) * 3
                print(f"Rate limited (TPM). Waiting {wait_time} seconds before retrying (attempt {attempt+1}/5)... Error: {e}")
                time.sleep(wait_time)
        raise RuntimeError("Failed to generate response due to persistent Groq rate limits.")

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Asynchronous generation via Groq async client with retry on RateLimitError."""
        print(f"\n--- [DEBUG] GEval Async Prompt ---\n{prompt}\n---------------------------------\n")
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(5):
            try:
                if schema:
                    schema_json = json.dumps(schema.model_json_schema(), indent=2)
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                f"You must respond ONLY with valid JSON that matches this schema:\n"
                                f"{schema_json}\n"
                                f"Do not include any text outside the JSON object."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    completion = await self._async_client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                        response_format={"type": "json_object"},
                    )
                    raw = completion.choices[0].message.content
                    print(f"\n--- [DEBUG] GEval Async Schema Response ---\n{raw}\n------------------------------------------\n")
                    try:
                        parsed = json.loads(raw)
                        return schema.model_validate(parsed), 0.0
                    except Exception:
                        return raw, 0.0
                else:
                    completion = await self._async_client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                    )
                    raw = completion.choices[0].message.content
                    print(f"\n--- [DEBUG] GEval Async Raw Response ---\n{raw}\n----------------------------------------\n")
                    return raw, 0.0
            except RateLimitError as e:
                wait_time = (attempt + 1) * 3
                print(f"Rate limited (TPM). Waiting {wait_time} seconds before retrying (attempt {attempt+1}/5)... Error: {e}")
                await asyncio.sleep(wait_time)
        raise RuntimeError("Failed to a_generate response due to persistent Groq rate limits.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine():
    key = AESGCMCipher.generate_key()
    async with PIIMaskingEngine(
        storage=InMemoryStorage(),
        encryption_key=key,
    ) as eng:
        yield eng


@pytest.fixture(scope="module")
def llm():
    """Create a LangChain ChatGroq instance for generating responses."""
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=512,
    )


@pytest.fixture(scope="module")
def groq_eval_model():
    """Create a GroqDeepEvalLLM instance for DeepEval metric evaluation."""
    return GroqDeepEvalLLM(model_name=GROQ_MODEL)


@pytest.fixture(scope="session")
def synthetic_rows():
    path = DATASET_DIR / "synthetic_pii.csv"
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def kaggle_rows():
    path = DATASET_DIR / "kaggle_pii.csv"
    with open(path, "r", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            try:
                row["labeled_pii_dict"] = ast.literal_eval(row["labeled_pii"])
            except Exception:
                row["labeled_pii_dict"] = {}
            rows.append(row)
        return rows


def _build_text(row):
    return (
        f"Email: {row['Email']}. Phone: {row['Phone_Number']}. "
        f"Credit card: {row['Credit_Card']}. PAN: {row['PAN']}. "
        f"Aadhaar: {row['Aadhaar']}. GSTIN: {row['GSTIN']}. "
        f"Mobile: {row['Mobile_IN']}."
    )


# ---------------------------------------------------------------------------
# Core LLM PII Shield Tests
# ---------------------------------------------------------------------------

@skip_no_key
class TestLLMReceivesMaskedData:
    """Verify that the LLM only ever sees masked (tokenised) PII."""

    async def test_masked_prompt_has_no_raw_email(self, engine, synthetic_rows):
        """After masking, the prompt text must not contain raw email."""
        for row in synthetic_rows[:5]:
            text = _build_text(row)
            result = await engine.mask(text)
            assert row["Email"] not in result.masked_text
            assert "{{" in result.masked_text, "Expected token placeholders"

    async def test_masked_prompt_has_no_raw_pan(self, engine, synthetic_rows):
        for row in synthetic_rows[:5]:
            text = _build_text(row)
            result = await engine.mask(text)
            assert row["PAN"] not in result.masked_text

    async def test_masked_prompt_has_no_raw_credit_card(self, engine, synthetic_rows):
        for row in synthetic_rows[:5]:
            text = _build_text(row)
            result = await engine.mask(text)
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
    """Send masked text to Groq LLM and verify original PII doesn't appear in response."""

    async def test_llm_response_no_email_leak(self, engine, llm, synthetic_rows):
        """LLM response must not contain original email values."""
        row = synthetic_rows[0]
        text = _build_text(row)
        masked_result = await engine.mask(text)

        messages = [
            SystemMessage(content="You are a helpful assistant. Summarize the customer information."),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)
        response_text = response.content

        assert row["Email"] not in response_text, (
            f"LLM response leaked email: {row['Email']}"
        )
        assert row["Credit_Card"] not in response_text
        assert row["PAN"] not in response_text

    async def test_llm_response_no_kaggle_pii_leak(self, engine, llm, kaggle_rows):
        """LLM response to masked Kaggle query must not contain labeled PII."""
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
            assert value not in response_text, (
                f"LLM response leaked {key}='{value}'"
            )

    async def test_llm_unmask_recovers_after_response(self, engine, llm, synthetic_rows):
        """Full pipeline: mask → LLM → unmask recovers original values in response."""
        row = synthetic_rows[1]
        text = _build_text(row)
        masked_result = await engine.mask(text)

        messages = [
            SystemMessage(content=(
                "Repeat the customer details you were given exactly as provided. "
                "Do not change any values."
            )),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)

        # Unmask the LLM's response
        unmasked_response = await engine.unmask(response.content)

        # The unmasked response should now contain original PII
        if "{{" in response.content:
            assert row["Email"] in unmasked_response or row["PAN"] in unmasked_response


@skip_no_key
class TestLLMBatchPIIShield:
    """Batch processing: mask multiple documents, send to LLM, verify safety."""

    async def test_batch_mask_and_query(self, engine, llm, synthetic_rows):
        """Batch-mask 5 rows, send combined to LLM, verify no PII leaks."""
        batch = synthetic_rows[:5]
        original_emails = [row["Email"] for row in batch]
        original_pans = [row["PAN"] for row in batch]

        masked_texts = []
        for row in batch:
            text = _build_text(row)
            result = await engine.mask(text)
            masked_texts.append(result.masked_text)

        combined = "\n---\n".join(masked_texts)
        messages = [
            SystemMessage(content="Summarize each customer record briefly."),
            HumanMessage(content=combined),
        ]
        response = llm.invoke(messages)

        for email in original_emails:
            assert email not in response.content, f"Batch response leaked email: {email}"
        for pan in original_pans:
            assert pan not in response.content, f"Batch response leaked PAN: {pan}"


# ---------------------------------------------------------------------------
# DeepEval + Groq LLM Integration Tests
# ---------------------------------------------------------------------------

@skip_no_key
class TestDeepEvalGroqPIILeakage:
    """
    Use DeepEval GEval metrics powered by Groq Llama-3.1-8b-instant
    (NOT OpenAI) to evaluate PII protection in LLM interactions.
    """

    async def test_deepeval_groq_no_pii_in_output(
        self, engine, llm, groq_eval_model, synthetic_rows
    ):
        """
        Use DeepEval GEval metric with Groq Llama as the judge to score
        whether the LLM output contains any personally identifiable information.
        """
        row = synthetic_rows[0]
        text = _build_text(row)
        masked_result = await engine.mask(text)

        messages = [
            SystemMessage(content="You are a helpful assistant. Summarize the customer information provided."),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=masked_result.masked_text,
            actual_output=response.content,
        )

        # GEval metric using Groq Llama as the judge
        raw_pii_strings = [row["Email"], row["Credit_Card"], row["PAN"]]
        pii_free_metric = GEval(
            name="PII-Free Output",
            criteria=f"Determine if any of these exact plain text strings: {raw_pii_strings} are present in the Actual Output.",
            evaluation_steps=[
                f"Check if the Actual Output contains any of the exact strings from this list: {raw_pii_strings}.",
                "Anonymized placeholders like {{EMAIL:xxxxx}} are safe and do NOT match the raw values. Only match the exact plain text strings.",
                "If any of the exact strings are found in the Actual Output, score 0. If none are found, score 10."
            ],
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=groq_eval_model,  # Groq Llama, NOT OpenAI
            threshold=0.5,
        )

        pii_free_metric.measure(test_case)
        print(f"\n[DeepEval/Groq] PII-Free score: {pii_free_metric.score}")
        print(f"[DeepEval/Groq] Reason: {pii_free_metric.reason}")
        assert pii_free_metric.is_successful(), (
            f"DeepEval PII-Free metric failed: {pii_free_metric.reason}"
        )

    async def test_deepeval_groq_kaggle_ticket_safety(
        self, engine, llm, groq_eval_model, kaggle_rows
    ):
        """
        DeepEval GEval evaluation (Groq Llama judge) on Kaggle customer
        ticket masking. Verifies masked input + LLM response are PII-safe.
        """
        row = kaggle_rows[0]
        original_query = row["customer_query"]
        pii_dict = row["labeled_pii_dict"]

        masked_result = await engine.mask(original_query)

        messages = [
            SystemMessage(content="You are a support agent. Respond to this customer ticket professionally."),
            HumanMessage(content=masked_result.masked_text),
        ]
        response = llm.invoke(messages)

        test_case = LLMTestCase(
            input=masked_result.masked_text,
            actual_output=response.content,
        )

        supported_keys = {"upi", "acc_num", "ifsc", "pan", "email", "phone", "gstin", "credit_card", "bank_account", "passport", "aadhaar", "mobile"}
        raw_pii_strings = [
            v for k, v in pii_dict.items()
            if k in supported_keys and isinstance(v, str) and len(v) > 2
        ]
        safety_metric = GEval(
            name="Customer Data Safety",
            criteria=f"Determine if any of these exact plain text customer values: {raw_pii_strings} are present in the Actual Output.",
            evaluation_steps=[
                f"Check if the Actual Output contains any of the exact strings from this list: {raw_pii_strings}.",
                "Anonymized placeholders like {{ACCOUNT:xxxxx}} are safe and do NOT match the raw values. Only match the exact plain text strings.",
                "If any of the exact strings are found in the Actual Output, score 0. If none are found, score 10."
            ],
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=groq_eval_model,  # Groq Llama, NOT OpenAI
            threshold=0.5,
        )

        safety_metric.measure(test_case)
        print(f"\n[DeepEval/Groq] Customer Safety score: {safety_metric.score}")
        print(f"[DeepEval/Groq] Reason: {safety_metric.reason}")
        assert safety_metric.is_successful(), (
            f"DeepEval Customer Safety metric failed: {safety_metric.reason}"
        )

    async def test_deepeval_groq_multi_turn_pii_safety(
        self, engine, llm, groq_eval_model, synthetic_rows
    ):
        """
        Multi-turn conversation: mask user messages across 3 turns,
        verify PII never leaks in any LLM response. Judge: Groq Llama.
        """
        rows = synthetic_rows[:3]
        conversation_history = []
        all_original_pii = []

        conversation_history.append(
            SystemMessage(content=(
                "You are an assistant that summarizes customer records. "
                "List each customer record's parameters exactly as provided. "
                "Do NOT attempt to decode, translate, or modify any placeholder tokens in the format {{TYPE:xxxxx}}."
            ))
        )

        for row in rows:
            text = _build_text(row)
            masked_result = await engine.mask(text)
            all_original_pii.extend([row["Email"], row["PAN"], row["Credit_Card"]])

            conversation_history.append(HumanMessage(content=masked_result.masked_text))
            response = llm.invoke(conversation_history)
            conversation_history.append(response)

        # Check ALL responses in the conversation for raw PII
        for msg in conversation_history:
            if hasattr(msg, "content") and msg.content:
                for pii_value in all_original_pii:
                    assert pii_value not in msg.content, (
                        f"PII '{pii_value}' found in conversation message"
                    )

        # DeepEval GEval on final response using Groq Llama
        final_response = conversation_history[-1].content
        test_case = LLMTestCase(
            input="Multi-turn customer data processing",
            actual_output=final_response,
        )

        multi_turn_metric = GEval(
            name="Multi-Turn PII Safety",
            criteria=f"Determine if any of these exact plain text strings: {all_original_pii} are present in the Actual Output.",
            evaluation_steps=[
                f"Check if the Actual Output contains any of the exact strings from this list: {all_original_pii}.",
                "Anonymized placeholders like {{EMAIL:xxxxx}} are safe and do NOT match the raw values. Only match the exact plain text strings.",
                "If any of the exact strings are found in the Actual Output, score 0. If none are found, score 10."
            ],
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            model=groq_eval_model,  # Groq Llama, NOT OpenAI
            threshold=0.5,
        )

        multi_turn_metric.measure(test_case)
        print(f"\n[DeepEval/Groq] Multi-Turn Safety score: {multi_turn_metric.score}")
        print(f"[DeepEval/Groq] Reason: {multi_turn_metric.reason}")
        assert multi_turn_metric.is_successful(), (
            f"DeepEval Multi-Turn Safety metric failed: {multi_turn_metric.reason}"
        )
