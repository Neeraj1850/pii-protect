"""Integration tests for async context manager lifecycle.

Tests cover:
- async with properly initialises and closes engine
- Operations after close raise EngineNotInitialisedError
- Multiple sequential context manager uses
- Nested context managers with different storage
"""
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine, EngineNotInitialisedError
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestContextManager:
    async def test_async_with_lifecycle(self):
        key = AESGCMCipher.generate_key()
        async with PIIMaskingEngine(
            storage=InMemoryStorage(), encryption_key=key
        ) as engine:
            result = await engine.mask("john@acme.com")
            assert result.token_count >= 1

    async def test_operations_after_close_raise(self):
        key = AESGCMCipher.generate_key()
        engine = PIIMaskingEngine(storage=InMemoryStorage(), encryption_key=key)
        async with engine:
            await engine.mask("test@test.com")
        with pytest.raises(EngineNotInitialisedError):
            await engine.mask("test@test.com")

    async def test_sequential_context_managers(self):
        key = AESGCMCipher.generate_key()
        storage = InMemoryStorage()
        async with PIIMaskingEngine(storage=storage, encryption_key=key) as e1:
            r1 = await e1.mask("john@acme.com")
        async with PIIMaskingEngine(storage=storage, encryption_key=key) as e2:
            restored = await e2.unmask(r1.masked_text)
            assert "john@acme.com" in restored

    async def test_different_storage_isolation(self):
        key = AESGCMCipher.generate_key()
        async with PIIMaskingEngine(
            storage=InMemoryStorage(), encryption_key=key
        ) as e1:
            r1 = await e1.mask("john@acme.com")
        async with PIIMaskingEngine(
            storage=InMemoryStorage(), encryption_key=key
        ) as e2:
            result = await e2.unmask(r1.masked_text)
            assert "UNRESOLVED" in result or "john@acme.com" not in result
