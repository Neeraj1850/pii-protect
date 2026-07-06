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

```
.
├── pyproject.toml              # Project dependencies & pytest configuration
├── run_tests.sh                # Executable shell script runner for all suites
├── README.md                   # Repository documentation
├── reports/                    # Generated test & coverage reports (HTML)
└── tests/
    ├── conftest.py             # Global test fixtures (Engine, Storage, PII samples)
    ├── unit/
    │   ├── test_crypto.py          # AES-256-GCM cipher verification & IV freshness
    │   ├── test_ner_engine.py      # Regex detection layers & span conflict resolution
    │   ├── test_redact.py          # Irreversible masking (sync, no storage write)
    │   ├── test_storage_memory.py  # InMemoryStorage key/value/scope operations
    │   ├── test_storage_filesystem.py# FileSystemStorage file persistence & atomic writes
    │   └── test_error_handling.py  # Exceptions, missing dependency extras
    ├── integration/
    │   ├── test_mask_unmask_flow.py# Full masking/unmasking roundtrip validation
    │   ├── test_scope_deduplication.py# Token deduplication within context scopes
    │   ├── test_mask_dict.py      # JSON/Dict recursive string masking
    │   ├── test_context_manager.py # Async context manager lifecycle validation
    │   └── test_encryption_key.py  # Key management (explicit key vs ephemeral)
    ├── llm/
    │   ├── test_llm_prompt_masking.py# System & User prompt isolation before LLM query
    │   ├── test_llm_response_unmask.py# Restoring variables from LLM echoed responses
    │   └── test_llm_batch_processing.py# Concurrency, batching, and document RAG pipelines
    ├── detection/
    │   ├── test_email_phone.py     # Email formats, country-specific phone formatting
    │   ├── test_financial_ids.py   # GST, PAN, TAN, IBAN, SWIFT, Credit Cards
    │   └── test_mixed_pii.py       # Multi-PII complex business document contexts
    ├── edge_cases/
    │   └── test_edge_cases.py      # Empty inputs, large text, unicode & emoji boundary tests
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
