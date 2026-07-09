"""
Fixtures for live Groq API tests: real LangChain + Groq calls, and a
DeepEval GEval judge also powered by Groq (never OpenAI).

Requires GROQ_API_KEY in .env / the environment. Tests using these
fixtures are marked pytest.mark.llm_integration and skip individually
(via skip_no_key) rather than erroring when the key is absent, so the
rest of the suite stays runnable without network access.
"""
import json
import os
import time
import asyncio
from typing import Optional, Tuple, Union

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq

from groq import Groq, AsyncGroq, RateLimitError

from deepeval.models import DeepEvalBaseLLM

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"


def _has_groq_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


skip_no_key = pytest.mark.skipif(not _has_groq_key(), reason="GROQ_API_KEY not set")


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEval model wrapper that routes evaluation calls to Groq's llama-3.1-8b-instant."""

    def __init__(self, model_name: str = GROQ_MODEL):
        self._model_name = model_name
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        # Do NOT call super().__init__() — DeepEvalBaseLLM.__init__ calls
        # load_model() and sets self.model; we handle that manually below.
        self.name = model_name
        self.model = self._client

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return self._model_name

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Synchronous generation via Groq chat completions, retrying on rate limits."""
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
                    try:
                        return schema.model_validate(json.loads(raw)), 0.0
                    except Exception:
                        return raw, 0.0
                else:
                    completion = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                    )
                    return completion.choices[0].message.content, 0.0
            except RateLimitError:
                time.sleep((attempt + 1) * 3)
        raise RuntimeError("Failed to generate response due to persistent Groq rate limits.")

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Asynchronous generation via Groq's async client, retrying on rate limits."""
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
                    try:
                        return schema.model_validate(json.loads(raw)), 0.0
                    except Exception:
                        return raw, 0.0
                else:
                    completion = await self._async_client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=1024,
                    )
                    return completion.choices[0].message.content, 0.0
            except RateLimitError:
                await asyncio.sleep((attempt + 1) * 3)
        raise RuntimeError("Failed to a_generate response due to persistent Groq rate limits.")


@pytest.fixture(scope="module")
def llm():
    """A LangChain ChatGroq instance for generating responses."""
    return ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=512)


@pytest.fixture(scope="module")
def groq_eval_model():
    """A GroqDeepEvalLLM instance for DeepEval GEval metric evaluation."""
    return GroqDeepEvalLLM(model_name=GROQ_MODEL)


@pytest.fixture(autouse=True)
def _time_groq_calls(request):
    """Print wall-clock time for every real Groq API call made during a
    live test: ChatGroq.invoke() (the assistant response) and
    GroqDeepEvalLLM.generate()/a_generate() (the DeepEval GEval judge).
    Only patches fixtures the test actually requested, so tests that don't
    use `llm`/`groq_eval_model` don't pay for constructing them.
    """
    patched = []

    def _wrap(obj, attr_name, label):
        original = getattr(obj, attr_name)

        if asyncio.iscoroutinefunction(original):
            # Calling an `async def` only creates a coroutine object — it
            # doesn't run the body until awaited. Timing the call itself
            # (as the sync branch below does) would measure ~0s regardless
            # of how long the real network call takes, so this branch must
            # await the coroutine inside the timer instead.
            async def timed(*args, **kwargs):
                start = time.perf_counter()
                result = await original(*args, **kwargs)
                elapsed = time.perf_counter() - start
                print(f"\n[groq call] {label}.{attr_name} took {elapsed:.2f}s")
                return result
        else:
            def timed(*args, **kwargs):
                start = time.perf_counter()
                result = original(*args, **kwargs)
                elapsed = time.perf_counter() - start
                print(f"\n[groq call] {label}.{attr_name} took {elapsed:.2f}s")
                return result

        # ChatGroq is a pydantic model and rejects setattr() for names that
        # aren't declared fields; bypass validation via object.__setattr__.
        object.__setattr__(obj, attr_name, timed)
        patched.append((obj, attr_name, original))

    if "llm" in request.fixturenames:
        _wrap(request.getfixturevalue("llm"), "invoke", "ChatGroq")

    if "groq_eval_model" in request.fixturenames:
        eval_model = request.getfixturevalue("groq_eval_model")
        _wrap(eval_model, "generate", "GroqDeepEvalLLM")
        _wrap(eval_model, "a_generate", "GroqDeepEvalLLM")

    yield

    for obj, attr_name, original in patched:
        object.__setattr__(obj, attr_name, original)
