"""Unit tests for RedisStorage (pii_protect.storage.redis_backend).

Requires a reachable Redis instance (see tests/unit/storage/conftest.py for
how the URL is resolved / how tests skip when unavailable). Tests run
against a dedicated DB index (15 by default) under a per-test key prefix,
so they never touch application data in DB 0.

Tests cover:
- put/get roundtrip, get_nonexistent, get_many, get_many_partial
- put() idempotency (existing key is not overwritten)
- find_by_value_hash with and without scope, and scope isolation
- touch() increments access_count (starts at 0, unlike Postgres which
  inserts with access_count=1)
- TTL expiry when ttl_seconds is set
- key_prefix isolation between two RedisStorage instances
- Using the backend before connect() raises RuntimeError
"""
import asyncio

import pytest
import pytest_asyncio

from pii_protect.storage.redis_backend import RedisStorage
from pii_protect.types import TokenRecord

from .conftest import REDIS_URL

pytestmark = [pytest.mark.unit, pytest.mark.requires_redis, pytest.mark.asyncio]


def make_record(token="{{EMAIL:rd001}}", value_hash="rdhash1", scope=None):
    return TokenRecord(
        token_value=token,
        entity_type="EMAIL",
        ciphertext=b"encrypted_rd",
        iv=b"iv_rd_data__",
        tag=b"tag_rd_data_",
        original_length=13,
        value_hash=value_hash,
        scope=scope,
    )


class TestRedisStorage:
    async def test_put_and_get_roundtrip(self, redis_storage):
        record = make_record()
        await redis_storage.put(record)
        result = await redis_storage.get(record.token_value)
        assert result is not None
        assert result.token_value == record.token_value
        assert result.ciphertext == record.ciphertext
        assert result.iv == record.iv
        assert result.tag == record.tag

    async def test_get_nonexistent_returns_none(self, redis_storage):
        result = await redis_storage.get("{{FAKE:00000}}")
        assert result is None

    async def test_put_does_not_overwrite_existing(self, redis_storage):
        record = make_record(token="{{EMAIL:rd002}}", value_hash="h2")
        await redis_storage.put(record)

        clashing = make_record(token="{{EMAIL:rd002}}", value_hash="different-hash")
        await redis_storage.put(clashing)

        result = await redis_storage.get("{{EMAIL:rd002}}")
        assert result.value_hash == "h2"  # first write wins

    async def test_get_many(self, redis_storage):
        r1 = make_record(token="{{EMAIL:rd010}}", value_hash="h10")
        r2 = make_record(token="{{PHONE:rd020}}", value_hash="h20")
        await redis_storage.put(r1)
        await redis_storage.put(r2)
        results = await redis_storage.get_many([r1.token_value, r2.token_value])
        assert len(results) == 2

    async def test_get_many_partial(self, redis_storage):
        r1 = make_record(token="{{EMAIL:rd030}}", value_hash="h30")
        await redis_storage.put(r1)
        results = await redis_storage.get_many([r1.token_value, "{{FAKE:rd999}}"])
        assert len(results) == 1
        assert r1.token_value in results

    async def test_get_many_empty_list(self, redis_storage):
        assert await redis_storage.get_many([]) == {}

    async def test_find_by_value_hash(self, redis_storage):
        record = make_record(value_hash="rd_unique", scope="scope-1")
        await redis_storage.put(record)
        found = await redis_storage.find_by_value_hash("rd_unique", scope="scope-1")
        assert found == record.token_value

    async def test_find_by_value_hash_not_found(self, redis_storage):
        result = await redis_storage.find_by_value_hash("nonexistent", scope=None)
        assert result is None

    async def test_find_by_value_hash_none_scope(self, redis_storage):
        record = make_record(token="{{EMAIL:rd040}}", value_hash="h40", scope=None)
        await redis_storage.put(record)
        found = await redis_storage.find_by_value_hash("h40", scope=None)
        assert found == record.token_value

    async def test_find_by_value_hash_scope_isolation(self, redis_storage):
        record = make_record(token="{{EMAIL:rd050}}", value_hash="h50", scope="doc-a")
        await redis_storage.put(record)
        found = await redis_storage.find_by_value_hash("h50", scope="doc-b")
        assert found is None

    async def test_touch_increments_access_count(self, redis_storage):
        record = make_record(token="{{EMAIL:rd060}}", value_hash="h60")
        await redis_storage.put(record)
        first = await redis_storage.get(record.token_value)
        assert first.access_count == 0  # Redis respects the record's initial value

        await redis_storage.touch(record.token_value)
        second = await redis_storage.get(record.token_value)
        assert second.access_count == 1

    async def test_ttl_expiry(self):
        """A record stored with a short TTL disappears after it elapses."""
        storage = RedisStorage(url=REDIS_URL, key_prefix="pii_test_ttl:", ttl_seconds=1)
        try:
            await storage.connect()
        except Exception as exc:
            pytest.skip(f"Redis not reachable: {exc}")

        try:
            record = make_record(token="{{EMAIL:rdttl}}", value_hash="httl")
            await storage.put(record)
            assert await storage.get(record.token_value) is not None

            await asyncio.sleep(1.5)
            assert await storage.get(record.token_value) is None
        finally:
            await storage.close()

    async def test_key_prefix_isolation(self):
        """Two RedisStorage instances with different prefixes never collide."""
        s1 = RedisStorage(url=REDIS_URL, key_prefix="pii_test_iso_a:")
        s2 = RedisStorage(url=REDIS_URL, key_prefix="pii_test_iso_b:")
        await s1.connect()
        await s2.connect()
        try:
            record = make_record(token="{{EMAIL:rdiso}}", value_hash="hiso")
            await s1.put(record)
            assert await s2.get(record.token_value) is None
        finally:
            for s in (s1, s2):
                client = s._client
                keys = [k async for k in client.scan_iter(match=f"{s._prefix}*")]
                if keys:
                    await client.delete(*keys)
                await s.close()

    async def test_use_before_connect_raises(self):
        storage = RedisStorage(url="redis://unused")
        with pytest.raises(RuntimeError):
            storage._require_client()
