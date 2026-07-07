"""Integration tests: PIIMaskingEngine end-to-end against real RedisStorage.

Mirrors tests/integration/test_mask_unmask_flow.py but proves the full
mask -> encrypt -> persist -> unmask -> decrypt pipeline also holds when
the vault is a real Redis instance rather than InMemoryStorage.

Tests cover:
- mask() persists tokens that a real Redis-backed unmask() can resolve
- Deduplication (repeated value -> same token) survives a real backend
- Same value under different scopes creates separate Redis dedup entries
- redact() never touches storage (no keys written even though mask() would)
- Data survives across two engine instances sharing the same key prefix
  (simulating separate app processes sharing one Redis instance)
- Concurrent mask() calls against the shared connection pool
"""
import asyncio
import re

import pytest
import pytest_asyncio

from pii_protect import PIIMaskingEngine

pytestmark = [pytest.mark.integration, pytest.mark.requires_redis, pytest.mark.asyncio]


class TestRedisEngineRoundtrip:
    async def test_mask_unmask_roundtrip(self, redis_engine):
        text = "Contact john@acme.com about GST 27AAPFU0939F1ZV"
        masked = await redis_engine.mask(text)
        assert "john@acme.com" not in masked.masked_text
        assert "27AAPFU0939F1ZV" not in masked.masked_text

        restored = await redis_engine.unmask(masked.masked_text)
        assert restored == text

    async def test_repeated_value_dedupes_to_same_token(self, redis_engine):
        text = "CC john@acme.com and BCC john@acme.com"
        result = await redis_engine.mask(text)
        tokens = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", result.masked_text)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]

    async def test_same_value_different_scopes_shares_one_entry(self, redis_engine):
        """
        Token generation is scope-independent (same value + entity_type + salt
        -> same token, see DeterministicTokenGenerator), so masking the same
        value under two different scopes produces the identical token_value.
        RedisStorage.put() is a no-op when the token key already exists, so
        the second scope's write never happens — only one Redis hash exists
        for this value. unmask() still works for both scopes, since it
        resolves purely by token_value, not scope.
        """
        masked_a = await redis_engine.mask("Email john@acme.com", scope="doc-a")
        masked_b = await redis_engine.mask("Email john@acme.com", scope="doc-b")
        assert masked_a.masked_text == masked_b.masked_text

        client = redis_engine._storage._client
        prefix = redis_engine._storage._prefix
        token_keys = [k async for k in client.scan_iter(match=f"{prefix}token:*")]
        assert len(token_keys) == 1

        assert (await redis_engine.unmask(masked_a.masked_text)) == "Email john@acme.com"
        assert (await redis_engine.unmask(masked_b.masked_text)) == "Email john@acme.com"

    async def test_redact_writes_nothing_to_storage(self, redis_engine):
        redis_engine.redact("Contact john@acme.com")
        client = redis_engine._storage._client
        prefix = redis_engine._storage._prefix
        keys = [k async for k in client.scan_iter(match=f"{prefix}*")]
        assert keys == []

    async def test_data_persists_across_engine_instances_same_prefix(self, redis_engine):
        """A second engine pointed at the same key prefix can unmask the first's tokens."""
        original = "Contact john@acme.com"
        masked = await redis_engine.mask(original)

        from pii_protect.storage.redis_backend import RedisStorage

        second_storage = RedisStorage(
            url=redis_engine._storage._url, key_prefix=redis_engine._storage._prefix
        )
        second_engine = PIIMaskingEngine(
            storage=second_storage, encryption_key=redis_engine._cipher.key
        )
        await second_engine.initialise()
        try:
            restored = await second_engine.unmask(masked.masked_text)
            assert restored == original
        finally:
            await second_engine.close()

    async def test_concurrent_mask_calls(self, redis_engine):
        texts = [f"Email user{i}@acme.com" for i in range(10)]
        results = await asyncio.gather(*(redis_engine.mask(t) for t in texts))
        for t, result in zip(texts, results):
            restored = await redis_engine.unmask(result.masked_text)
            assert restored == t
