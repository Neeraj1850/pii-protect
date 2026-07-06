"""PII detection tests for financial identifiers."""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.detection, pytest.mark.asyncio]


class TestIndianFinancialIDs:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    @pytest.mark.parametrize("gst", [
        "27AAPFU0939F1ZV",
        "29GGGGG1314R9Z6",
        "07AAACH7409R1ZZ",
    ])
    async def test_detects_gst(self, engine, gst):
        result = await engine.mask(f"GST number: {gst}")
        assert gst not in result.masked_text
        assert "GST" in result.entity_counts

    @pytest.mark.parametrize("pan", ["ABCDE1234F", "ZZZZZ9999Z"])
    async def test_detects_pan(self, engine, pan):
        result = await engine.mask(f"PAN: {pan}")
        assert pan not in result.masked_text
        assert "PAN" in result.entity_counts

    @pytest.mark.parametrize("tan", ["DELA12345F", "MUMM99999B"])
    async def test_detects_tan(self, engine, tan):
        result = await engine.mask(f"TAN: {tan}")
        assert tan not in result.masked_text

    async def test_multiple_indian_ids(self, engine):
        text = "GST 27AAPFU0939F1ZV, PAN ABCDE1234F, TAN DELA12345F"
        result = await engine.mask(text)
        assert "27AAPFU0939F1ZV" not in result.masked_text
        assert "ABCDE1234F" not in result.masked_text
        assert "DELA12345F" not in result.masked_text
        assert result.token_count >= 3


class TestInternationalFinancialIDs:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    @pytest.mark.parametrize("iban", [
        "GB29NWBK60161331926819",
        "DE89370400440532013000",
        "FR7630006000011234567890189",
    ])
    async def test_detects_iban(self, engine, iban):
        result = await engine.mask(f"IBAN: {iban}")
        assert iban not in result.masked_text
        assert "IBAN" in result.entity_counts

    @pytest.mark.parametrize("swift", ["DEUTDEFF", "NWBKGB2L", "BNPAFRPP"])
    async def test_detects_swift(self, engine, swift):
        result = await engine.mask(f"SWIFT: {swift}")
        assert swift not in result.masked_text


class TestCreditCards:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    @pytest.mark.parametrize("cc,card_type", [
        ("4111111111111111", "Visa"),
        ("5500000000000004", "Mastercard"),
        ("340000000000009", "Amex"),
    ])
    async def test_detects_credit_cards(self, engine, cc, card_type):
        result = await engine.mask(f"Card ({card_type}): {cc}")
        assert cc not in result.masked_text
        assert result.token_count >= 1

    async def test_credit_card_with_dashes(self, engine):
        result = await engine.mask("Card: 4111-1111-1111-1111")
        assert "4111-1111-1111-1111" not in result.masked_text


class TestInvoicePO:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_detects_invoice_number(self, engine):
        result = await engine.mask("Invoice: INV-2026-00417")
        assert result.token_count >= 1

    async def test_detects_po_number(self, engine):
        result = await engine.mask("PO Reference: PO-2026-0042")
        assert result.token_count >= 1

    async def test_invoice_and_po_in_context(self, engine):
        text = (
            "Re: Invoice INV-2026-00417 for PO-2026-0042. "
            "Amount: 5,00,000. Contact billing@acme.com."
        )
        result = await engine.mask(text)
        assert "billing@acme.com" not in result.masked_text
        assert result.token_count >= 2
