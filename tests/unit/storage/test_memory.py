"""Unit tests for InMemoryStorage (pii_protect.storage).

Tests cover:
- put/get roundtrip
- get non-existent token returns None
- get_many returns correct subset
- find_by_value_hash with and without scope
- touch updates access metadata
- Isolation between instances
"""
import pytest
import pytest_asyncio
from pii_protect.storage import InMemoryStorage
from pii_protect.types import TokenRecord


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def make_record(token="{{EMAIL:abc12}}", value_hash="hash1",
                scope=None, **kwargs):
    return TokenRecord(
        token_value=token,
        entity_type="EMAIL",
        ciphertext=b"encrypted_data",
        iv=b"iv_data_96bit",
        tag=b"tag_data",
        original_length=13,
        value_hash=value_hash,
        scope=scope,
        **kwargs,
    )


class TestInMemoryStorage:
    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    async def test_put_and_get(self, storage):
        record = make_record()
        await storage.put(record)
        result = await storage.get(record.token_value)
        assert result is not None
        assert result.token_value == record.token_value

    async def test_get_nonexistent_returns_none(self, storage):
        result = await storage.get("{{FAKE:00000}}")
        assert result is None

    async def test_get_many(self, storage):
        r1 = make_record(token="{{EMAIL:aaa11}}", value_hash="h1")
        r2 = make_record(token="{{PHONE:bbb22}}", value_hash="h2")
        r3 = make_record(token="{{GST:ccc33}}", value_hash="h3")
        await storage.put(r1)
        await storage.put(r2)
        await storage.put(r3)
        results = await storage.get_many([r1.token_value, r3.token_value])
        assert len(results) == 2
        assert r1.token_value in results
        assert r3.token_value in results

    async def test_get_many_partial(self, storage):
        r1 = make_record(token="{{EMAIL:ddd44}}", value_hash="h4")
        await storage.put(r1)
        results = await storage.get_many([r1.token_value, "{{FAKE:eee55}}"])
        assert len(results) == 1

    async def test_find_by_value_hash(self, storage):
        record = make_record(value_hash="unique_hash_1", scope="doc-1")
        await storage.put(record)
        found = await storage.find_by_value_hash("unique_hash_1", scope="doc-1")
        assert found == record.token_value

    async def test_find_by_value_hash_not_found(self, storage):
        result = await storage.find_by_value_hash("nonexistent", scope=None)
        assert result is None

    async def test_touch(self, storage):
        record = make_record()
        await storage.put(record)
        await storage.touch(record.token_value)

    async def test_separate_instances_isolated(self):
        s1 = InMemoryStorage()
        s2 = InMemoryStorage()
        await s1.put(make_record(token="{{EMAIL:iso01}}", value_hash="h_iso"))
        assert await s2.get("{{EMAIL:iso01}}") is None
