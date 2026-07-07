"""
Shared fixtures for real-backend storage tests (PostgreSQL, Redis).

Both fixtures connect to a *local* instance by default (matching the
dev/CI environment this suite ships with) and can be pointed elsewhere via
TEST_POSTGRES_DSN / TEST_REDIS_URL. If the backend isn't reachable, tests
are skipped rather than failed, so the rest of the suite stays runnable
on machines without Postgres/Redis installed.

Each test gets an isolated namespace (a unique Postgres schema / Redis key
prefix + dedicated DB index) so tests never see each other's data and never
touch anything outside their sandbox.
"""
import uuid

import pytest
import pytest_asyncio

from tests._backend_config import POSTGRES_DSN, REDIS_URL


def _unique_schema() -> str:
    """Postgres unquoted identifiers can't start with a digit, so prefix it."""
    return f"pii_test_{uuid.uuid4().hex[:12]}"


def _unique_prefix() -> str:
    return f"pii_test:{uuid.uuid4().hex[:12]}:"


@pytest_asyncio.fixture
async def pg_storage():
    """A connected PostgresStorage in a throwaway schema, dropped on teardown."""
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        pytest.skip("asyncpg not installed — pip install 'pii-protect[postgres]'")

    from pii_protect.storage.postgres import PostgresStorage

    schema = _unique_schema()
    storage = PostgresStorage(POSTGRES_DSN, schema=schema)
    try:
        await storage.connect()
    except Exception as exc:  # noqa: BLE001 - any connection failure means "skip"
        pytest.skip(f"PostgreSQL not reachable at {POSTGRES_DSN}: {exc}")

    yield storage

    await storage.close()
    cleanup_conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        await cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        await cleanup_conn.close()


@pytest_asyncio.fixture
async def redis_storage():
    """A connected RedisStorage under a throwaway key prefix, flushed on teardown."""
    try:
        from redis import asyncio as redis_asyncio  # noqa: F401
    except ImportError:
        pytest.skip("redis not installed — pip install 'pii-protect[redis]'")

    from pii_protect.storage.redis_backend import RedisStorage

    prefix = _unique_prefix()
    storage = RedisStorage(url=REDIS_URL, key_prefix=prefix)
    try:
        await storage.connect()
    except Exception as exc:  # noqa: BLE001 - any connection failure means "skip"
        pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")

    yield storage

    client = storage._client
    keys = [key async for key in client.scan_iter(match=f"{prefix}*")]
    if keys:
        await client.delete(*keys)
    await storage.close()
