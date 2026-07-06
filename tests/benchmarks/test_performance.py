"""Performance benchmark tests for pii-protect.

Uses pytest-benchmark to measure performance of key operations.
Run with: pytest tests/benchmarks/ -v --benchmark-only
"""
import asyncio
import pytest
import pytest_asyncio
from pii_protect import PIIMaskingEngine, NEREngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher


pytestmark = [pytest.mark.benchmark, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helper to run async in benchmark
# ---------------------------------------------------------------------------
def run_async(coro):
    """Run an async coroutine synchronously for benchmarking."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Benchmark fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def benchmark_key():
    return AESGCMCipher.generate_key()


@pytest.fixture(scope="module")
def benchmark_cipher(benchmark_key):
    return AESGCMCipher(benchmark_key)


@pytest.fixture(scope="module")
def benchmark_ner():
    return NEREngine()


# ---------------------------------------------------------------------------
# Crypto benchmarks
# ---------------------------------------------------------------------------
class TestCryptoBenchmarks:
    def test_key_generation_speed(self, benchmark):
        benchmark(AESGCMCipher.generate_key)

    def test_encrypt_short_text(self, benchmark, benchmark_cipher):
        benchmark(benchmark_cipher.encrypt, "john@acme.com", "EMAIL")

    def test_encrypt_medium_text(self, benchmark, benchmark_cipher):
        text = "A" * 1000
        benchmark(benchmark_cipher.encrypt, text, "TEXT")

    def test_encrypt_decrypt_roundtrip(self, benchmark, benchmark_cipher):
        def roundtrip():
            enc = benchmark_cipher.encrypt("john@acme.com", "EMAIL")
            benchmark_cipher.decrypt(enc, "EMAIL")
        benchmark(roundtrip)


# ---------------------------------------------------------------------------
# NER detection benchmarks
# ---------------------------------------------------------------------------
class TestNERBenchmarks:
    def test_detect_short_text(self, benchmark, benchmark_ner):
        benchmark(benchmark_ner.detect, "Contact john@acme.com about GST 27AAPFU0939F1ZV")

    def test_detect_no_pii(self, benchmark, benchmark_ner):
        benchmark(benchmark_ner.detect, "The weather is nice today with clear skies.")

    def test_detect_dense_pii(self, benchmark, benchmark_ner):
        text = (
            "Email john@acme.com, GST 27AAPFU0939F1ZV, PAN ABCDE1234F, "
            "call +91 98765 43210, IBAN GB29NWBK60161331926819"
        )
        benchmark(benchmark_ner.detect, text)

    def test_detect_large_text(self, benchmark, benchmark_ner):
        paragraphs = []
        for i in range(10):
            paragraphs.append(
                f"Section {i}: Contact person{i}@company.com for details. "
                f"GST reference 27AAPFU0939F1ZV. "
                "This is filler text to make the document longer. " * 5
            )
        text = "\n\n".join(paragraphs)
        benchmark(benchmark_ner.detect, text)


# ---------------------------------------------------------------------------
# End-to-end benchmarks
# ---------------------------------------------------------------------------
class TestEndToEndBenchmarks:
    def test_mask_short_text(self, benchmark, benchmark_key):
        async def do_mask():
            async with PIIMaskingEngine(
                storage=InMemoryStorage(), encryption_key=benchmark_key
            ) as engine:
                await engine.mask("Contact john@acme.com about GST 27AAPFU0939F1ZV")

        benchmark(run_async, do_mask())

    def test_mask_unmask_roundtrip(self, benchmark, benchmark_key):
        async def do_roundtrip():
            async with PIIMaskingEngine(
                storage=InMemoryStorage(), encryption_key=benchmark_key
            ) as engine:
                result = await engine.mask("Contact john@acme.com about GST 27AAPFU0939F1ZV")
                await engine.unmask(result.masked_text)

        benchmark(run_async, do_roundtrip())

    def test_redact_short_text(self, benchmark, benchmark_key):
        async def do_redact():
            async with PIIMaskingEngine(
                storage=InMemoryStorage(), encryption_key=benchmark_key
            ) as engine:
                engine.redact("Contact john@acme.com about GST 27AAPFU0939F1ZV")

        benchmark(run_async, do_redact())

    def test_mask_dict_benchmark(self, benchmark, benchmark_key):
        async def do_mask_dict():
            async with PIIMaskingEngine(
                storage=InMemoryStorage(), encryption_key=benchmark_key
            ) as engine:
                data = {
                    "email": "john@acme.com",
                    "gst": "27AAPFU0939F1ZV",
                    "phone": "+91 98765 43210",
                    "pan": "ABCDE1234F",
                }
                await engine.mask_dict(data)

        benchmark(run_async, do_mask_dict())
