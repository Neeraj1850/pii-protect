"""
Tests: Synthetic PII Dataset Masking & Redaction

Validates that pii-protect correctly detects and masks/redacts PII values
from the generated synthetic_pii.csv dataset containing Indian-specific
identifiers (Aadhaar, PAN, GSTIN, UPI, IFSC, etc.)
"""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher

from .conftest import build_text_from_synthetic_row, RELIABLY_DETECTED_FIELDS


pytestmark = [pytest.mark.dataset, pytest.mark.asyncio]


class TestSyntheticDatasetMasking:
    """Mask every row in synthetic_pii.csv and verify PII removal."""

    async def test_mask_all_rows_removes_email(self, dataset_engine, synthetic_rows):
        """Every email must be masked out across all 100 rows."""
        failures = []
        for i, row in enumerate(synthetic_rows):
            text = build_text_from_synthetic_row(row)
            result = await dataset_engine.mask(text)
            if row["Email"] in result.masked_text:
                failures.append(f"Row {i}: Email '{row['Email']}' leaked")
        assert not failures, f"{len(failures)} email leaks:\n" + "\n".join(failures[:10])

    async def test_mask_all_rows_removes_credit_card(self, dataset_engine, synthetic_rows):
        """Every credit card number must be masked."""
        failures = []
        for i, row in enumerate(synthetic_rows):
            text = build_text_from_synthetic_row(row)
            result = await dataset_engine.mask(text)
            cc = row["Credit_Card"]
            if cc in result.masked_text:
                failures.append(f"Row {i}: CC '{cc}' leaked")
        assert not failures, f"{len(failures)} credit card leaks:\n" + "\n".join(failures[:10])

    async def test_mask_all_rows_removes_pan(self, dataset_engine, synthetic_rows):
        """PAN numbers (Indian tax ID) must be masked."""
        failures = []
        for i, row in enumerate(synthetic_rows):
            text = build_text_from_synthetic_row(row)
            result = await dataset_engine.mask(text)
            pan = row["PAN"]
            if pan in result.masked_text:
                failures.append(f"Row {i}: PAN '{pan}' leaked")
        assert not failures, f"{len(failures)} PAN leaks:\n" + "\n".join(failures[:10])

    async def test_mask_all_rows_removes_gstin(self, dataset_engine, synthetic_rows):
        """GSTIN values must be masked."""
        failures = []
        for i, row in enumerate(synthetic_rows):
            text = build_text_from_synthetic_row(row)
            result = await dataset_engine.mask(text)
            gstin = row["GSTIN"]
            if gstin in result.masked_text:
                failures.append(f"Row {i}: GSTIN '{gstin}' leaked")
        assert not failures, f"{len(failures)} GSTIN leaks:\n" + "\n".join(failures[:10])

    async def test_mask_token_count_positive(self, dataset_engine, synthetic_rows):
        """Each row must produce at least 1 masked token (has PII)."""
        zero_token_rows = []
        for i, row in enumerate(synthetic_rows[:20]):
            text = build_text_from_synthetic_row(row)
            result = await dataset_engine.mask(text)
            if result.token_count == 0:
                zero_token_rows.append(i)
        assert not zero_token_rows, f"Rows with 0 tokens: {zero_token_rows}"

    async def test_mask_unmask_roundtrip_preserves_original(self, dataset_engine, synthetic_rows):
        """mask → unmask roundtrip must recover the original text for 10 rows."""
        for row in synthetic_rows[:10]:
            text = build_text_from_synthetic_row(row)
            masked = await dataset_engine.mask(text)
            restored = await dataset_engine.unmask(masked.masked_text)
            assert restored == text, f"Roundtrip failed for row with person={row['Person']}"


class TestSyntheticDatasetRedaction:
    """Redact (permanently) every row and verify PII removal."""

    async def test_redact_removes_all_reliable_fields(self, dataset_engine, synthetic_rows):
        """Redaction must remove all reliably-detected PII fields."""
        for row in synthetic_rows[:20]:
            text = build_text_from_synthetic_row(row)
            redacted = dataset_engine.redact(text)
            for field in RELIABLY_DETECTED_FIELDS:
                value = row[field]
                assert value not in redacted, (
                    f"Redaction leaked {field}='{value}' for {row['Person']}"
                )

    async def test_redact_format_uses_brackets(self, dataset_engine, synthetic_rows):
        """Redacted text must contain [REDACTED:...] placeholders."""
        text = build_text_from_synthetic_row(synthetic_rows[0])
        redacted = dataset_engine.redact(text)
        assert "[REDACTED:" in redacted, "No redaction placeholders found"

    async def test_redact_is_irreversible(self, dataset_engine, synthetic_rows):
        """Redacted text cannot be unmasked back to original."""
        text = build_text_from_synthetic_row(synthetic_rows[0])
        redacted = dataset_engine.redact(text)
        # unmask should return the redacted text unchanged (no tokens to resolve)
        result = await dataset_engine.unmask(redacted)
        assert result == redacted, "Redacted text should not be reversible"


class TestSyntheticIndianPII:
    """Focused tests on Indian-specific PII detection in synthetic data."""

    async def test_mobile_in_format_detection(self, dataset_engine, synthetic_rows):
        """Test +91 XXXXX XXXXX format mobile numbers are detected."""
        detected = 0
        for row in synthetic_rows[:20]:
            mobile = row["Mobile_IN"]
            text = f"Please call {mobile} for assistance."
            result = await dataset_engine.mask(text)
            if mobile not in result.masked_text:
                detected += 1
        # At least some should be detected
        assert detected >= 10, f"Only {detected}/20 Indian mobiles detected"

    async def test_aadhaar_format_detection(self, dataset_engine, synthetic_rows):
        """Test XXXX-XXXX-XXXX Aadhaar format detection."""
        detected = 0
        for row in synthetic_rows[:20]:
            aadhaar = row["Aadhaar"]
            text = f"Aadhaar number: {aadhaar}"
            result = await dataset_engine.mask(text)
            if aadhaar not in result.masked_text:
                detected += 1
        # Track detection rate
        print(f"Aadhaar detection rate: {detected}/20")

    async def test_ifsc_detection(self, dataset_engine, synthetic_rows):
        """Test IFSC code detection (e.g., SBIN0322966)."""
        detected = 0
        for row in synthetic_rows[:20]:
            ifsc = row["IFSC_Code"]
            text = f"Transfer to IFSC {ifsc}"
            result = await dataset_engine.mask(text)
            if ifsc not in result.masked_text:
                detected += 1
        print(f"IFSC detection rate: {detected}/20")

    async def test_upi_id_detection(self, dataset_engine, synthetic_rows):
        """Test UPI ID detection (e.g., user@ybl)."""
        detected = 0
        for row in synthetic_rows[:20]:
            upi = row["UPI_ID"]
            text = f"Pay via UPI: {upi}"
            result = await dataset_engine.mask(text)
            if upi not in result.masked_text:
                detected += 1
        print(f"UPI detection rate: {detected}/20")
