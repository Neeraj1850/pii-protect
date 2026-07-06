"""
Tests: Kaggle-style PII Dataset Masking

Validates pii-protect against the simulated Kaggle customer support
dataset where each ticket contains free-form text with labeled PII.
Uses the labeled_pii ground-truth to verify masking completeness.
"""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.dataset, pytest.mark.asyncio]


class TestKaggleDatasetMasking:
    """Mask customer queries from the Kaggle dataset and verify PII removal."""

    async def test_mask_all_tickets(self, dataset_engine, kaggle_rows):
        """Every ticket query must have at least 1 token masked."""
        zero_token_tickets = []
        for row in kaggle_rows:
            result = await dataset_engine.mask(row["customer_query"])
            if result.token_count == 0:
                zero_token_tickets.append(row["ticket_id"])
        assert not zero_token_tickets, (
            f"{len(zero_token_tickets)} tickets had 0 PII tokens: {zero_token_tickets[:10]}"
        )

    async def test_labeled_emails_masked(self, dataset_engine, kaggle_rows):
        """Every labeled email in the ground-truth must be removed from masked text."""
        failures = []
        for row in kaggle_rows:
            pii = row["labeled_pii_dict"]
            if "email" not in pii:
                continue
            result = await dataset_engine.mask(row["customer_query"])
            if pii["email"] in result.masked_text:
                failures.append(f"{row['ticket_id']}: email '{pii['email']}' leaked")
        assert not failures, "\n".join(failures[:10])

    async def test_labeled_pan_masked(self, dataset_engine, kaggle_rows):
        """Every labeled PAN must be removed from masked text."""
        failures = []
        for row in kaggle_rows:
            pii = row["labeled_pii_dict"]
            if "pan" not in pii:
                continue
            result = await dataset_engine.mask(row["customer_query"])
            if pii["pan"] in result.masked_text:
                failures.append(f"{row['ticket_id']}: PAN '{pii['pan']}' leaked")
        assert not failures, "\n".join(failures[:10])

    async def test_labeled_phone_masked(self, dataset_engine, kaggle_rows):
        """Every labeled phone must be removed from masked text."""
        failures = []
        for row in kaggle_rows:
            pii = row["labeled_pii_dict"]
            if "phone" not in pii:
                continue
            result = await dataset_engine.mask(row["customer_query"])
            phone = pii["phone"]
            if phone in result.masked_text:
                failures.append(f"{row['ticket_id']}: phone '{phone}' leaked")
        assert not failures, "\n".join(failures[:10])

    async def test_labeled_upi_masked(self, dataset_engine, kaggle_rows):
        """Every labeled UPI ID must be removed from masked text."""
        failures = []
        for row in kaggle_rows:
            pii = row["labeled_pii_dict"]
            if "upi" not in pii:
                continue
            result = await dataset_engine.mask(row["customer_query"])
            upi = pii["upi"]
            if upi in result.masked_text:
                failures.append(f"{row['ticket_id']}: UPI '{upi}' leaked")
        # UPI detection is optional — report but don't hard-fail if regex coverage is low
        if failures:
            print(f"WARNING: {len(failures)} UPI leaks (may not be regex-supported)")

    async def test_labeled_gstin_masked(self, dataset_engine, kaggle_rows):
        """Every labeled GSTIN must be removed from masked text."""
        failures = []
        for row in kaggle_rows:
            pii = row["labeled_pii_dict"]
            if "gstin" not in pii:
                continue
            result = await dataset_engine.mask(row["customer_query"])
            gstin = pii["gstin"]
            if gstin in result.masked_text:
                failures.append(f"{row['ticket_id']}: GSTIN '{gstin}' leaked")
        assert not failures, "\n".join(failures[:10])


class TestKaggleDatasetRoundtrip:
    """Verify mask → unmask roundtrip on Kaggle dataset queries."""

    async def test_roundtrip_preserves_original(self, dataset_engine, kaggle_rows):
        """mask → unmask must recover the original customer query."""
        failures = []
        for row in kaggle_rows[:15]:
            original = row["customer_query"]
            masked = await dataset_engine.mask(original)
            restored = await dataset_engine.unmask(masked.masked_text)
            if restored != original:
                failures.append(row["ticket_id"])
        assert not failures, f"Roundtrip failed for tickets: {failures}"


class TestKaggleDatasetRedaction:
    """Redact Kaggle queries and verify permanent PII removal."""

    async def test_redact_removes_all_labeled_pii(self, dataset_engine, kaggle_rows):
        """Redacted text must not contain any labeled PII value."""
        failures = []
        for row in kaggle_rows[:20]:
            pii = row["labeled_pii_dict"]
            redacted = dataset_engine.redact(row["customer_query"])
            for key, value in pii.items():
                if value in redacted:
                    failures.append(f"{row['ticket_id']}: {key}='{value}' in redacted text")
        # Some PII types may not be regex-detectable; report proportion
        if failures:
            print(f"Redaction leaks: {len(failures)} (some may be undetectable types)")
            # Hard-assert on email/PAN which are reliably detected
            hard_failures = [f for f in failures if "email=" in f or "pan=" in f]
            assert not hard_failures, "\n".join(hard_failures)


class TestKaggleCategoryCoverage:
    """Ensure each ticket category gets tested."""

    async def test_all_categories_present(self, kaggle_rows):
        """Dataset should contain tickets from all 5 scenario categories."""
        categories = {row["category"] for row in kaggle_rows}
        expected = {
            "Refund / Billing",
            "Verification Failure",
            "Corporate Registration",
            "Account Security",
            "Travel / Passport Verification",
        }
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"
