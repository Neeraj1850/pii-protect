"""
Fixtures for integration tests that exercise PIIMaskingEngine against real
Postgres/Redis backends (as opposed to InMemoryStorage/FileSystemStorage
used everywhere else in tests/integration/).

See tests/_backend_config.py for how the DSN/URL is resolved, and
tests/unit/storage/conftest.py for the equivalent bare-storage fixtures.
"""
import uuid

import pytest
import pytest_asyncio

from pii_protect import PIIMaskingEngine
from pii_protect.crypto import AESGCMCipher

from tests._backend_config import POSTGRES_DSN, REDIS_URL


@pytest_asyncio.fixture
async def pg_engine():
    """A PIIMaskingEngine backed by a real PostgresStorage in a throwaway schema."""
    try:
        import asyncpg
    except ImportError:
        pytest.skip("asyncpg not installed — pip install 'pii-protect[postgres]'")

    from pii_protect.storage.postgres import PostgresStorage

    schema = f"pii_test_{uuid.uuid4().hex[:12]}"
    storage = PostgresStorage(POSTGRES_DSN, schema=schema)
    engine = PIIMaskingEngine(storage=storage, encryption_key=AESGCMCipher.generate_key())
    try:
        await engine.initialise()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not reachable at {POSTGRES_DSN}: {exc}")

    yield engine

    await engine.close()
    cleanup_conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        await cleanup_conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        await cleanup_conn.close()


@pytest_asyncio.fixture
async def redis_engine():
    """A PIIMaskingEngine backed by a real RedisStorage under a throwaway key prefix."""
    try:
        from redis import asyncio as redis_asyncio  # noqa: F401
    except ImportError:
        pytest.skip("redis not installed — pip install 'pii-protect[redis]'")

    from pii_protect.storage.redis_backend import RedisStorage

    prefix = f"pii_test:{uuid.uuid4().hex[:12]}:"
    storage = RedisStorage(url=REDIS_URL, key_prefix=prefix)
    engine = PIIMaskingEngine(storage=storage, encryption_key=AESGCMCipher.generate_key())
    try:
        await engine.initialise()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")

    yield engine

    client = storage._client
    keys = [key async for key in client.scan_iter(match=f"{prefix}*")]
    if keys:
        await client.delete(*keys)
    await engine.close()
