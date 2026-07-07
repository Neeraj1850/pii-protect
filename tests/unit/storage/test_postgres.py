"""Unit tests for PostgresStorage (pii_protect.storage.postgres).

Requires a reachable PostgreSQL instance (see tests/unit/storage/conftest.py
for how the DSN is resolved / how tests skip when unavailable).

Tests cover:
- Schema/table auto-creation on connect()
- put/get roundtrip, get_nonexistent, get_many
- put() idempotency (ON CONFLICT DO NOTHING)
- find_by_value_hash with and without scope, and scope isolation
- touch() increments access_count (Postgres inserts access_count=1 on put,
  unlike the in-memory/filesystem backends which start at 0)
- log_access() writes an audit_log row
- Using the backend before connect() raises RuntimeError
"""
import pytest
import pytest_asyncio

from pii_protect.storage.postgres import PostgresStorage
from pii_protect.types import TokenRecord

pytestmark = [pytest.mark.unit, pytest.mark.requires_postgres, pytest.mark.asyncio]


def make_record(token="{{EMAIL:pg001}}", value_hash="pghash1", scope=None):
    return TokenRecord(
        token_value=token,
        entity_type="EMAIL",
        ciphertext=b"encrypted_pg",
        iv=b"iv_pg_data__",
        tag=b"tag_pg_data_",
        original_length=13,
        value_hash=value_hash,
        scope=scope,
    )


class TestPostgresStorage:
    async def test_connect_creates_schema_and_tables(self, pg_storage):
        pool = pg_storage._pool
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = $1 AND table_name = 'token_map')",
                pg_storage._schema,
            )
        assert exists is True

    async def test_put_and_get_roundtrip(self, pg_storage):
        record = make_record()
        await pg_storage.put(record)
        result = await pg_storage.get(record.token_value)
        assert result is not None
        assert result.token_value == record.token_value
        assert result.ciphertext == record.ciphertext
        assert result.iv == record.iv
        assert result.tag == record.tag

    async def test_get_nonexistent_returns_none(self, pg_storage):
        result = await pg_storage.get("{{FAKE:00000}}")
        assert result is None

    async def test_put_is_idempotent(self, pg_storage):
        record = make_record(token="{{EMAIL:pg002}}", value_hash="pghash2")
        await pg_storage.put(record)
        await pg_storage.put(record)  # second insert must be a no-op
        pool = pg_storage._pool
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {pg_storage._schema}.token_map WHERE token_value = $1",
                record.token_value,
            )
        assert count == 1

    async def test_get_many(self, pg_storage):
        r1 = make_record(token="{{EMAIL:pg010}}", value_hash="h10")
        r2 = make_record(token="{{PHONE:pg020}}", value_hash="h20")
        await pg_storage.put(r1)
        await pg_storage.put(r2)
        results = await pg_storage.get_many([r1.token_value, r2.token_value, "{{FAKE:99999}}"])
        assert len(results) == 2
        assert r1.token_value in results
        assert r2.token_value in results

    async def test_get_many_empty_list(self, pg_storage):
        assert await pg_storage.get_many([]) == {}

    async def test_find_by_value_hash(self, pg_storage):
        record = make_record(value_hash="pg_unique", scope="scope-1")
        await pg_storage.put(record)
        found = await pg_storage.find_by_value_hash("pg_unique", scope="scope-1")
        assert found == record.token_value

    async def test_find_by_value_hash_not_found(self, pg_storage):
        result = await pg_storage.find_by_value_hash("nonexistent", scope=None)
        assert result is None

    async def test_find_by_value_hash_none_scope(self, pg_storage):
        record = make_record(token="{{EMAIL:pg030}}", value_hash="h30", scope=None)
        await pg_storage.put(record)
        found = await pg_storage.find_by_value_hash("h30", scope=None)
        assert found == record.token_value

    async def test_find_by_value_hash_scope_isolation(self, pg_storage):
        """Same value_hash under a different scope must not be found."""
        record = make_record(token="{{EMAIL:pg040}}", value_hash="h40", scope="doc-a")
        await pg_storage.put(record)
        found = await pg_storage.find_by_value_hash("h40", scope="doc-b")
        assert found is None

    async def test_touch_increments_access_count(self, pg_storage):
        record = make_record(token="{{EMAIL:pg050}}", value_hash="h50")
        await pg_storage.put(record)
        first = await pg_storage.get(record.token_value)
        assert first.access_count == 1  # Postgres inserts with access_count=1

        await pg_storage.touch(record.token_value)
        second = await pg_storage.get(record.token_value)
        assert second.access_count == 2

    async def test_log_access_writes_audit_row(self, pg_storage):
        record = make_record(token="{{EMAIL:pg060}}", value_hash="h60", scope="doc-audit")
        await pg_storage.put(record)
        await pg_storage.log_access(record.token_value, "MASK", "test-actor", scope="doc-audit")

        pool = pg_storage._pool
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT operation, actor, scope FROM {pg_storage._schema}.access_log "
                f"WHERE token_value = $1",
                record.token_value,
            )
        assert row is not None
        assert row["operation"] == "MASK"
        assert row["actor"] == "test-actor"
        assert row["scope"] == "doc-audit"

    async def test_use_before_connect_raises(self):
        storage = PostgresStorage("postgresql://unused/unused")
        with pytest.raises(RuntimeError):
            storage._require_pool()

    async def test_custom_schema_isolation(self, pg_storage):
        """Records in one schema are invisible to a differently-scoped storage."""
        record = make_record(token="{{EMAIL:pg070}}", value_hash="h70")
        await pg_storage.put(record)

        other = PostgresStorage(pg_storage._dsn, schema="pii_test_other_" + pg_storage._schema[-8:])
        await other.connect()
        try:
            assert await other.get(record.token_value) is None
        finally:
            pool = other._pool
            async with pool.acquire() as conn:
                await conn.execute(f"DROP SCHEMA IF EXISTS {other._schema} CASCADE")
            await other.close()
