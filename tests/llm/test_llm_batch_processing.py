"""LLM-specific tests: batch processing for production LLM pipelines.

Tests cover:
- Processing multiple documents concurrently
- Consistent masking across a batch
- Unmask after batch processing
- Large document masking performance
- Mixed PII density documents
- Idempotent masking (masking already-masked text)
"""
import asyncio
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.llm, pytest.mark.asyncio]


class TestLLMBatchProcessing:
    @pytest_asyncio.fixture
    async def engine(self):
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
        ) as eng:
            yield eng

    async def test_batch_mask_multiple_docs(self, engine):
        docs = [
            "Email from user1@test.com about PAN ABCDE1234F",
            "Invoice sent to user2@test.com, GST 27AAPFU0939F1ZV",
            "Call +91 98765 43210 for details on PO-2026-001",
        ]
        results = await asyncio.gather(*[engine.mask(d) for d in docs])
        assert all(r.token_count >= 1 for r in results)

    async def test_batch_roundtrip(self, engine):
        docs = [f"Document {i}: contact user{i}@example.com" for i in range(20)]
        mask_results = await asyncio.gather(*[engine.mask(d) for d in docs])
        unmask_results = await asyncio.gather(
            *[engine.unmask(r.masked_text) for r in mask_results]
        )
        for orig, restored in zip(docs, unmask_results):
            assert orig == restored

    async def test_mixed_pii_density(self, engine):
        sparse = "The quarterly report shows growth in the tech sector."
        dense = (
            "From john@acme.com to jane@corp.com, CC admin@test.org. "
            "GST 27AAPFU0939F1ZV, PAN ABCDE1234F, call +91 98765 43210."
        )
        r_sparse = await engine.mask(sparse)
        r_dense = await engine.mask(dense)
        assert r_sparse.token_count == 0
        assert r_dense.token_count >= 5

    async def test_large_document_handling(self, engine):
        paragraphs = []
        for i in range(50):
            paragraphs.append(
                f"Section {i}: This is filler text for the document. "
                f"Contact person{i}@company.com for details about GST 27AAPFU0939F1ZV."
            )
        large_doc = "\n\n".join(paragraphs)
        result = await engine.mask(large_doc)
        assert result.token_count >= 50
        assert "person0@company.com" not in result.masked_text

    async def test_idempotent_masking(self, engine):
        original = "Contact john@acme.com"
        r1 = await engine.mask(original)
        r2 = await engine.mask(r1.masked_text)
        assert r2.token_count == 0
        assert r2.masked_text == r1.masked_text
