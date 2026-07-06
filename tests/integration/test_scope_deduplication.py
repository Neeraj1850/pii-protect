"""Integration tests for scope-based deduplication.

Tests cover:
- Same PII + same scope -> same token (deduplication)
- Same PII + different scope -> may produce different stored entries
- scope passed to mask() and unmask() must match
- None scope behavior
"""
import re
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestScopeDeduplication:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_same_scope_deduplicates(self, engine):
        scope = "invoice-2026-001"
        r1 = await engine.mask("john@acme.com", scope=scope)
        r2 = await engine.mask("Also john@acme.com", scope=scope)
        t1 = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", r1.masked_text)
        t2 = re.findall(r"\{\{EMAIL:[a-f0-9]+\}\}", r2.masked_text)
        assert t1[0] == t2[0]

    async def test_unmask_with_matching_scope(self, engine):
        scope = "doc-42"
        result = await engine.mask("Contact john@acme.com", scope=scope)
        restored = await engine.unmask(result.masked_text, scope=scope)
        assert "john@acme.com" in restored

    async def test_scope_none_works(self, engine):
        result = await engine.mask("john@acme.com", scope=None)
        restored = await engine.unmask(result.masked_text, scope=None)
        assert "john@acme.com" in restored

    async def test_multiple_scopes_independent(self, engine):
        s1, s2 = "scope-A", "scope-B"
        r1 = await engine.mask("john@acme.com", scope=s1)
        r2 = await engine.mask("john@acme.com", scope=s2)
        res1 = await engine.unmask(r1.masked_text, scope=s1)
        res2 = await engine.unmask(r2.masked_text, scope=s2)
        assert "john@acme.com" in res1
        assert "john@acme.com" in res2
