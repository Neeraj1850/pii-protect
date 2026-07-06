"""PII detection tests for email addresses and phone numbers."""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.detection, pytest.mark.asyncio]


class TestEmailDetection:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    @pytest.mark.parametrize("email", [
        "john@acme.com",
        "jane.doe@example.org",
        "admin+test@sub.domain.co.uk",
        "user123@gmail.com",
        "ceo@company.io",
        "firstname.lastname@company.com",
        "email@subdomain.example.com",
    ])
    async def test_detects_valid_emails(self, engine, email):
        result = await engine.mask(f"Contact {email} for info")
        assert email not in result.masked_text
        assert result.token_count >= 1
        assert "EMAIL" in result.entity_counts

    async def test_multiple_emails_in_text(self, engine):
        text = "CC john@acme.com and jane@corp.com, BCC admin@test.org"
        result = await engine.mask(text)
        assert "john@acme.com" not in result.masked_text
        assert "jane@corp.com" not in result.masked_text
        assert "admin@test.org" not in result.masked_text
        assert result.entity_counts.get("EMAIL", 0) >= 3

    async def test_email_in_sentence_context(self, engine):
        result = await engine.mask("Please reply to john@acme.com at your convenience.")
        assert "john@acme.com" not in result.masked_text
        assert "Please reply to" in result.masked_text
        assert "at your convenience." in result.masked_text


class TestPhoneDetection:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    @pytest.mark.parametrize("phone", [
        "+91 98765 43210",
        "+1-555-010-0000",
        "+44 20 7946 0958",
        "+1 (555) 012-3456",
    ])
    async def test_detects_international_phones(self, engine, phone):
        result = await engine.mask(f"Call {phone}")
        assert phone not in result.masked_text
        assert result.token_count >= 1

    async def test_phone_with_text_context(self, engine):
        result = await engine.mask("For support, dial +91 98765 43210 during business hours.")
        assert "+91 98765 43210" not in result.masked_text
        assert "For support" in result.masked_text
