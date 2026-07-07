# pii-protect Test Repository & LLM Evaluation Suite

This repository is designed to test the [`pii-protect`](https://pypi.org/project/pii-protect/) library comprehensively, with a focus on evaluating its capabilities as a PII shielding layer for Large Language Models (LLMs) and local data pipelines.

`pii-protect` is an on-premise-first, zero-server, pluggable library for Python that provides high-performance masking, unmasking, and permanent redaction of Personally Identifiable Information (PII) and sensitive business credentials.

---

## Architecture & Test Focus

The test suite validates all primary components of the library under various workloads:

```
                    ┌─────────────────────────────────────────┐
                    │             PIIMaskingEngine             │
                    │        (pii_protect.engine)               │
                    │                                           │
                    │   mask()    unmask()    redact()          │
                    └───────┬───────────┬───────────┬───────────┘
                            │           │           │
              ┌─────────────┘           │           └─── (Redaction unit tests)
              ▼                         ▼
   ┌─────────────────────┐   ┌─────────────────────┐   ┌───────────────────────────────┐
   │      NEREngine      │   │Token Generator      │   │        StorageBackend         │
   │  - RegexNERLayer    │   │  - Reversible tokens│   │  - InMemoryStorage            │
   │  - SpacyNERLayer    │   │  - Deduplication    │   │  - FileSystemStorage          │
   │  - PrivacyFilter    │   │  - Value Hash Map   │   │  - Postgres/Redis (Extras)    │
   └─────────────────────┘   └─────────────────────┘   └───────────────────────────────┘
```

---

## Project Structure

Tests are organised by *what* they exercise, going from smallest unit to full external integrations:

```
.
├── pyproject.toml              # Project dependencies & pytest configuration
├── run_tests.sh                # Executable shell script runner for all suites
├── README.md                   # Repository documentation
├── dataset/                    # Synthetic & Kaggle-style PII CSV fixtures
├── scripts/generate_datasets.py# Regenerates the CSV fixtures
└── tests/
    ├── conftest.py              # Global fixtures: Engine, Storage, PII samples, dataset rows
    ├── _backend_config.py       # DSN/URL resolution shared by Postgres/Redis fixtures
    │
    ├── unit/                    # Single component, no cross-component wiring
    │   ├── test_crypto.py           # AES-256-GCM cipher verification & IV freshness
    │   ├── test_tokens.py           # DeterministicTokenGenerator: format, determinism, parsing
    │   ├── test_ner_engine.py       # Regex detection layers & span conflict resolution
    │   ├── test_redact.py           # Irreversible masking (sync, no storage write)
    │   ├── test_error_handling.py   # Exceptions, missing dependency extras
    │   └── storage/                 # One file per StorageBackend implementation
    │       ├── test_memory.py           # InMemoryStorage
    │       ├── test_filesystem.py       # FileSystemStorage (atomic writes, persistence)
    │       ├── test_postgres.py         # PostgresStorage (schema, audit log, idempotency)
    │       └── test_redis.py            # RedisStorage (TTL, key-prefix isolation)
    │
    ├── integration/             # PIIMaskingEngine wired to a real backend, full flows
    │   ├── test_mask_unmask_flow.py     # Full masking/unmasking roundtrip validation
    │   ├── test_scope_deduplication.py  # Token deduplication within context scopes
    │   ├── test_mask_dict.py            # JSON/Dict recursive string masking
    │   ├── test_context_manager.py      # Async context manager lifecycle validation
    │   ├── test_encryption_key.py       # Key management (explicit key vs ephemeral)
    │   ├── test_postgres_engine.py      # Engine mask/unmask against a real PostgreSQL DB
    │   └── test_redis_engine.py         # Engine mask/unmask against a real Redis instance
    │
    ├── detection/                # PII entity detection accuracy
    │   ├── test_email_phone.py      # Email formats, country-specific phone formatting
    │   ├── test_financial_ids.py    # GST, PAN, TAN, IBAN, SWIFT, Credit Cards
    │   └── test_mixed_pii.py        # Multi-PII complex business document contexts
    │
    ├── edge_cases/
    │   └── test_edge_cases.py       # Empty inputs, large text, unicode & emoji boundary tests
    │
    ├── llm/                       # LLM-shaped masking behaviour
    │   ├── test_llm_prompt_masking.py     # System vs. user prompt isolation (no network)
    │   ├── test_llm_response_unmask.py    # Restoring variables from LLM-echoed responses
    │   ├── test_llm_batch_processing.py   # Concurrency, batching, document RAG pipelines
    │   └── live/                          # Real network calls to Groq (needs GROQ_API_KEY)
    │       ├── conftest.py                    # ChatGroq + GroqDeepEvalLLM judge fixtures
    │       ├── test_groq_masking_safety.py     # Masked prompts sent to Groq leak no raw PII
    │       └── test_deepeval_groq_metrics.py   # DeepEval GEval scoring, judged by Groq Llama
    │
    ├── dataset/                   # Bulk accuracy checks over the CSV fixtures (no network)
    │   ├── conftest.py                # dataset_engine, regex pattern extensions, row builders
    │   ├── test_kaggle_pii.py
    │   └── test_synthetic_pii.py
    │
    └── benchmarks/
        └── test_performance.py     # Crypto, NER, and engine end-to-end latency benchmarks
```

---

## Installation

Install the test suite along with the required extras. It is recommended to install with `[all]` to test every backend and detection layer:

```bash
# Clone the repository and navigate into it
cd pii-redact

# Install the library and test suite dependencies locally
pip install -e ".[dev]"
```

### Optional Extras for Testing:
If you want to test specific backends or NLP/Transformer components:
- `pip install "pii-protect[spacy]"` — Adds spaCy-based Named Entity Recognition (NER)
- `pip install "pii-protect[privacy-filter]"` — Adds local Transformer-based token classification
- `pip install "pii-protect[redis]"` — Adds Redis token vault storage
- `pip install "pii-protect[postgres]"` — Adds PostgreSQL database integration
- `pip install "pii-protect[all]"` — Installs all available backends, libraries, and dev tools.

If you test spaCy NER, make sure to download the model:
```bash
python -m spacy download en_core_web_sm
```

### Postgres / Redis backend tests

`tests/unit/storage/test_postgres.py`, `test_redis.py`, and
`tests/integration/test_postgres_engine.py` / `test_redis_engine.py` need a
reachable Postgres/Redis instance. They connect to a **local** instance by
default and each test runs in an isolated, throwaway schema (Postgres) or
key prefix + dedicated DB 15 (Redis), cleaned up automatically — they never
touch your app's own data.

- Default Postgres DSN: `postgresql://<your-os-user>@localhost:5432/postgres`
- Default Redis URL: `redis://localhost:6379/15`
- Override with `TEST_POSTGRES_DSN` / `TEST_REDIS_URL` env vars to point at a different instance.
- If neither is reachable, these tests are **skipped** (not failed) so the rest of the suite stays runnable without any infra installed.

### Live Groq LLM tests

`tests/llm/live/` makes real calls to the Groq API (LangChain `ChatGroq` +
a DeepEval `GEval` judge, both using `llama-3.1-8b-instant`) to prove that
masked prompts/responses never leak raw PII, end to end. Set
`GROQ_API_KEY` in `.env` (auto-loaded via `python-dotenv`) or the
environment; tests skip individually if it's absent. Free-tier Groq keys
have a tight tokens-per-minute limit, so running the full `live/` folder
back-to-back can hit `429` rate-limit errors — this is a quota limit, not a
test failure.

---

## Running the Tests

We provide a runner script `run_tests.sh` to simplify testing with automatic reporting and optional installation.

### 1. Run all test suites (Unit, Integration, LLM, Edge Cases, Detection)
```bash
./run_tests.sh
```

### 2. Install dependencies first, then run tests
```bash
./run_tests.sh --install
```

### 3. Run specific test groups
```bash
# Run unit tests only
./run_tests.sh --unit-only

# Run integration tests only
./run_tests.sh --integration-only

# Run LLM interaction tests only
./run_tests.sh --llm-only
```

### 4. Run performance latency benchmarks
```bash
./run_tests.sh --benchmark
```

### 5. Disable HTML reports or code coverage calculation
```bash
./run_tests.sh --no-html --no-cov
```

---

## Key Test Scenarios Validated

### 1. Reversible Masking (LLM Pre-processing)
Ensures sensitive details are replaced with random-looking, salted placeholders (e.g., `{{EMAIL:abcc2}}`) so prompts can be sent safely to third-party LLMs.

### 2. Reversible Unmasking (LLM Post-processing)
Ensures when an LLM replies echoing those placeholders, we can look up the encrypted vault and reconstruct the original text.

### 3. Permanent Redaction
Ensures logs and analytical data can be purged of PII permanently using the `redact()` method without making any storage/encryption calls.

### 4. Scope Deduplication
Ensures the same PII value within the same scope (e.g., a specific session or document ID) maps to the same token, preventing redundant keys and maintaining LLM context coherence.
