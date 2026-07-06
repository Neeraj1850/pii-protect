"""
Root conftest.py for pii-protect test suite.

Provides shared fixtures used across all test categories:
- Engine instances with various storage backends
- Encryption keys
- Sample PII data
- Async event loop configuration
"""

import sys
import importlib.util
import importlib.abc
import importlib.machinery
import importlib

class PiiShieldRedirectFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.startswith('pii_shield'):
            real_name = name.replace('pii_shield', 'pii_protect', 1)
            spec = importlib.util.find_spec(real_name)
            if spec:
                class RedirectLoader(importlib.abc.Loader):
                    def create_module(self, spec):
                        mod = importlib.import_module(real_name)
                        sys.modules[name] = mod
                        return mod
                    def exec_module(self, module):
                        pass
                return importlib.machinery.ModuleSpec(
                    name=name,
                    loader=RedirectLoader(),
                    is_package=spec.submodule_search_locations is not None
                )
        return None

sys.meta_path.insert(0, PiiShieldRedirectFinder())


import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from pii_protect import PIIMaskingEngine
from pii_protect.crypto import AESGCMCipher
from pii_protect.storage import InMemoryStorage, FileSystemStorage



# ---------------------------------------------------------------------------
# Encryption key fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def encryption_key() -> str:
    """Generate a stable encryption key for the entire test session."""
    return AESGCMCipher.generate_key()


@pytest.fixture
def fresh_encryption_key() -> str:
    """Generate a fresh encryption key per test (for isolation tests)."""
    return AESGCMCipher.generate_key()


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_storage() -> InMemoryStorage:
    """Provide a fresh InMemoryStorage instance per test."""
    return InMemoryStorage()


@pytest.fixture
def fs_storage_path(tmp_path: Path) -> Path:
    """Provide a temporary file path for FileSystemStorage."""
    return tmp_path / "test_vault.json"


@pytest.fixture
def fs_storage(fs_storage_path: Path) -> FileSystemStorage:
    """Provide a fresh FileSystemStorage instance per test."""
    return FileSystemStorage(str(fs_storage_path))


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine(memory_storage: InMemoryStorage, encryption_key: str) -> PIIMaskingEngine:
    """
    Provide an initialised PIIMaskingEngine with InMemoryStorage.
    Automatically handles init/close lifecycle via async context manager.
    """
    async with PIIMaskingEngine(
        storage=memory_storage,
        encryption_key=encryption_key,
    ) as eng:
        yield eng


@pytest_asyncio.fixture
async def fs_engine(fs_storage: FileSystemStorage, encryption_key: str) -> PIIMaskingEngine:
    """
    Provide an initialised PIIMaskingEngine with FileSystemStorage.
    """
    async with PIIMaskingEngine(
        storage=fs_storage,
        encryption_key=encryption_key,
    ) as eng:
        yield eng


@pytest_asyncio.fixture
async def engine_no_key(memory_storage: InMemoryStorage) -> PIIMaskingEngine:
    """
    Provide an engine without an explicit encryption key.
    Should generate an ephemeral key with a warning.
    """
    async with PIIMaskingEngine(storage=memory_storage) as eng:
        yield eng


# ---------------------------------------------------------------------------
# Sample PII data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_emails() -> list[str]:
    """Common email addresses for testing."""
    return [
        "john@acme.com",
        "jane.doe@example.org",
        "admin+test@sub.domain.co.uk",
        "user123@gmail.com",
        "ceo@company.io",
    ]


@pytest.fixture
def sample_phones() -> list[str]:
    """Phone numbers in various formats."""
    return [
        "+91 98765 43210",
        "+1-555-010-0000",
        "+44 20 7946 0958",
        "9876543210",
        "+1 (555) 012-3456",
    ]


@pytest.fixture
def sample_indian_ids() -> dict[str, list[str]]:
    """Indian financial identifiers: GST, PAN, TAN."""
    return {
        "GST": [
            "27AAPFU0939F1ZV",
            "29GGGGG1314R9Z6",
            "07AAACH7409R1ZZ",
        ],
        "PAN": [
            "ABCDE1234F",
            "ZZZZZ9999Z",
        ],
        "TAN": [
            "DELA12345F",
            "MUMM99999B",
        ],
    }


@pytest.fixture
def sample_international_ids() -> dict[str, list[str]]:
    """International financial identifiers."""
    return {
        "IBAN": [
            "GB29NWBK60161331926819",
            "DE89370400440532013000",
            "FR7630006000011234567890189",
        ],
        "SWIFT": [
            "DEUTDEFF",
            "NWBKGB2L",
            "BNPAFRPP",
        ],
    }


@pytest.fixture
def sample_credit_cards() -> list[str]:
    """Credit card numbers (test numbers)."""
    return [
        "4111111111111111",       # Visa
        "5500000000000004",       # Mastercard
        "340000000000009",        # Amex
        "4111-1111-1111-1111",    # Visa with dashes
    ]


@pytest.fixture
def sample_text_with_mixed_pii() -> str:
    """A paragraph containing multiple PII types."""
    return (
        "Dear Mr. Rajesh Kumar,\n\n"
        "Please find the invoice INV-2026-00417 for the services provided to "
        "Acme Corporation. The total amount of ₹5,00,000 has been credited to "
        "account number 12345678901234. Your GST number 27AAPFU0939F1ZV and PAN "
        "ABCDE1234F are on file. For any queries, contact us at billing@acme.com "
        "or call +91 98765 43210.\n\n"
        "Regards,\nFinance Department\nPO-2026-0042"
    )


@pytest.fixture
def sample_llm_prompt() -> str:
    """A realistic LLM prompt containing PII that should be masked."""
    return (
        "Summarise this customer complaint:\n\n"
        "From: priya.sharma@techcorp.in\n"
        "Subject: Invoice dispute - INV-2026-00512\n\n"
        "Hi,\nI am Priya Sharma from TechCorp Solutions (GST: 29GGGGG1314R9Z6). "
        "We received invoice INV-2026-00512 dated 01-Jul-2026 for ₹3,50,000 but "
        "the agreed amount was ₹2,75,000. Our PO reference is PO-2026-1187. "
        "Please contact me at +91 87654 32109 or priya.sharma@techcorp.in to "
        "resolve this. My PAN is BSRPS5432K.\n\n"
        "Thanks,\nPriya"
    )


@pytest.fixture
def sample_llm_system_prompt() -> str:
    """System prompt for LLM that should NOT be masked."""
    return (
        "You are a helpful customer service assistant. "
        "Analyse the following complaint and suggest a resolution. "
        "Be professional and empathetic."
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_pii_absent():
    """
    Returns a helper function that asserts no raw PII values
    appear in the given text.
    """
    def _check(text: str, pii_values: list[str]) -> None:
        for val in pii_values:
            assert val not in text, (
                f"PII value '{val}' was found in text but should have been masked/redacted"
            )
    return _check


@pytest.fixture
def assert_tokens_present():
    """
    Returns a helper function that asserts masked tokens
    ({{TYPE:xxxxx}} format) are present in the text.
    """
    import re
    def _check(text: str, expected_count: int | None = None) -> list[str]:
        tokens = re.findall(r"\{\{[A-Z_]+:[a-f0-9]+\}\}", text)
        if expected_count is not None:
            assert len(tokens) == expected_count, (
                f"Expected {expected_count} tokens but found {len(tokens)}: {tokens}"
            )
        assert len(tokens) > 0, "No masked tokens found in text"
        return tokens
    return _check


@pytest.fixture
def assert_redacted_present():
    """
    Returns a helper function that asserts [REDACTED:TYPE] markers
    are present in the text.
    """
    import re
    def _check(text: str, expected_count: int | None = None) -> list[str]:
        markers = re.findall(r"\[REDACTED:[A-Z_]+\]", text)
        if expected_count is not None:
            assert len(markers) == expected_count, (
                f"Expected {expected_count} redaction markers but found {len(markers)}: {markers}"
            )
        assert len(markers) > 0, "No redaction markers found in text"
        return markers
    return _check
