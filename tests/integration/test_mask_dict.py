"""Integration tests for mask_dict() and unmask_dict().

Tests cover:
- Masking all string leaf values in a dict
- Roundtrip mask_dict -> unmask_dict
- Nested dict masking
- Non-string values left untouched
- Same PII across fields maps to same token
- Empty dict
"""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestMaskDict:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_basic_dict_masking(self, engine):
        data = {"email": "john@acme.com", "gst": "27AAPFU0939F1ZV"}
        masked = await engine.mask_dict(data)
        assert "john@acme.com" not in str(masked)
        assert "27AAPFU0939F1ZV" not in str(masked)

    async def test_dict_roundtrip(self, engine):
        data = {
            "contact": "john@acme.com",
            "tax_id": "27AAPFU0939F1ZV",
            "phone": "+91 98765 43210",
        }
        masked = await engine.mask_dict(data)
        unmasked = await engine.unmask_dict(masked)
        assert unmasked["contact"] == data["contact"]
        assert unmasked["tax_id"] == data["tax_id"]

    async def test_nested_dict(self, engine):
        data = {
            "customer": {
                "email": "jane@example.com",
                "address": {"phone": "+1-555-0100"},
            },
            "order_id": "ORD-001",
        }
        masked = await engine.mask_dict(data)
        assert "jane@example.com" not in str(masked)

    async def test_non_string_values_unchanged(self, engine):
        data = {
            "email": "john@acme.com",
            "amount": 5000.00,
            "count": 42,
            "active": True,
            "tags": None,
        }
        masked = await engine.mask_dict(data)
        assert masked["amount"] == 5000.00
        assert masked["count"] == 42
        assert masked["active"] is True

    async def test_same_pii_across_fields_same_token(self, engine):
        data = {"sender": "john@acme.com", "cc": "john@acme.com"}
        masked = await engine.mask_dict(data)
        assert masked["sender"] == masked["cc"]

    async def test_empty_dict(self, engine):
        masked = await engine.mask_dict({})
        assert masked == {}

    async def test_list_values_in_dict(self, engine):
        data = {"emails": ["john@acme.com", "jane@acme.com"]}
        masked = await engine.mask_dict(data)
        masked_str = str(masked)
        assert "john@acme.com" not in masked_str
        assert "jane@acme.com" not in masked_str
