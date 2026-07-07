"""Integration tests: PIIMaskingEngine end-to-end against real PostgresStorage.

Mirrors tests/integration/test_mask_unmask_flow.py but proves the full
mask -> encrypt -> persist -> unmask -> decrypt pipeline also holds when
the vault is a real PostgreSQL database rather than InMemoryStorage.

Tests cover:
- mask() persists tokens that a real Postgres-backed unmask() can resolve
- Deduplication (repeated value -> same token) survives a real backend
- Scope isolation across mask/unmask calls
- redact() never touches storage (nothing written even though mask() would)
- Data survives across two engine instances sharing the same schema
  (simulating separate app processes/pods sharing one database)
"""
import pytest
import pytest_asyncio

from pii_protect import PIIMaskingEngine
from pii_protect.crypto import AESGCMCipher

pytestmark = [pytest.mark.integration, pytest.mark.requires_postgres, pytest.mark.asyncio]


class TestPostgresEngineRoundtrip:
    async def test_mask_unmask_roundtrip(self, pg_engine):
        text = "Contact john@acme.com about GST 27AAPFU0939F1ZV"
        masked = await pg_engine.mask(text)
        assert "john@acme.com" not in masked.masked_text
        assert "27AAPFU0939F1ZV" not in masked.masked_text

        restored = await pg_engine.unmask(masked.masked_text)
        assert restored == text

    async def test_repeated_value_dedupes_to_same_token(self, pg_engine):
        text = "CC john@acme.com and BCC john@acme.com"
        result = await pg_engine.mask(text)
        import re

        tokens = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", result.masked_text)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]

    async def test_same_value_different_scopes_shares_one_row(self, pg_engine):
        """
        Token generation is scope-independent (same value + entity_type + salt
        -> same token, see DeterministicTokenGenerator), so masking the same
        value under two different scopes produces the identical token_value.
        Postgres's put() is `ON CONFLICT (token_value) DO NOTHING`, so the
        second scope's write is silently dropped and only one row ever exists
        — tagged with whichever scope wrote first. unmask() still works for
        both, since it resolves purely by token_value, not scope.
        """
        masked_a = await pg_engine.mask("Email john@acme.com", scope="doc-a")
        masked_b = await pg_engine.mask("Email john@acme.com", scope="doc-b")
        assert masked_a.masked_text == masked_b.masked_text

        pool = pg_engine._storage._pool
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {pg_engine._storage._schema}.token_map"
            )
        assert count == 1
        assert (await pg_engine.unmask(masked_a.masked_text)) == "Email john@acme.com"
        assert (await pg_engine.unmask(masked_b.masked_text)) == "Email john@acme.com"

    async def test_redact_writes_nothing_to_storage(self, pg_engine):
        pg_engine.redact("Contact john@acme.com")
        pool = pg_engine._storage._pool
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {pg_engine._storage._schema}.token_map"
            )
        assert count == 0

    async def test_data_persists_across_engine_instances_same_schema(self, pg_engine):
        """A second engine pointed at the same schema can unmask the first's tokens."""
        original = "Contact john@acme.com"
        masked = await pg_engine.mask(original)

        from pii_protect.storage.postgres import PostgresStorage

        second_storage = PostgresStorage(
            pg_engine._storage._dsn, schema=pg_engine._storage._schema
        )
        second_engine = PIIMaskingEngine(
            storage=second_storage, encryption_key=pg_engine._cipher.key
        )
        await second_engine.initialise()
        try:
            restored = await second_engine.unmask(masked.masked_text)
            assert restored == original
        finally:
            await second_engine.close()

    async def test_unmask_result_reports_zero_unresolved(self, pg_engine):
        masked = await pg_engine.mask("Email john@acme.com")
        result = await pg_engine.unmask_with_stats(masked.masked_text)
        assert result.tokens_resolved == 1
        assert result.tokens_unresolved == 0
