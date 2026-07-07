"""
Performance benchmarks for pii-protect.

Uses pytest-benchmark to measure latency of every layer in the pipeline,
from cheapest to most expensive:

    Crypto (AES-256-GCM)  -->  Tokens  -->  NER detection  -->
    Storage backends (Memory / FileSystem / Postgres / Redis)  -->
    End-to-end engine (mask / unmask / redact / mask_dict)

Run everything:      pytest tests/benchmarks/ --benchmark-only
Run one group:       pytest tests/benchmarks/ --benchmark-only -k storage
Compare two runs:    pytest tests/benchmarks/ --benchmark-only --benchmark-compare

See docs/BENCHMARKS.md for a written breakdown of these results, including
the storage-backend comparison table.
"""
import itertools

import pytest

from pii_protect import PIIMaskingEngine, NEREngine
from pii_protect.storage import InMemoryStorage, FileSystemStorage
from pii_protect.crypto import AESGCMCipher
from pii_protect.tokens import DeterministicTokenGenerator
from pii_protect.types import EntityType, TokenRecord

from tests._backend_config import POSTGRES_DSN, REDIS_URL

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Helper to run async code inside a sync pytest-benchmark callable
# ---------------------------------------------------------------------------
_bench_loop = None


def run_async(coro):
    """
    Run an async coroutine synchronously for benchmarking.

    Reuses one lazily-created event loop across the whole module rather
    than a fresh loop per call: several fixtures here (Postgres pool,
    Redis client) create a connection bound to whichever loop was current
    when connect() ran, and a later call on a *different* loop breaks that
    connection. `asyncio.run()` tears its loop down after every call, so
    it can't be used here despite being the more modern one-liner.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()

    global _bench_loop
    if _bench_loop is None or _bench_loop.is_closed():
        _bench_loop = asyncio.new_event_loop()
    return _bench_loop.run_until_complete(coro)


def make_record(token: str, value_hash: str) -> TokenRecord:
    """A representative TokenRecord — same shape for every backend so put()/get()
    numbers are directly comparable across Memory/FileSystem/Postgres/Redis."""
    return TokenRecord(
        token_value=token,
        entity_type="EMAIL",
        ciphertext=b"0123456789abcdef",
        iv=b"iv_12_bytes_",
        tag=b"tag_16_bytes____",
        original_length=13,
        value_hash=value_hash,
    )


# ---------------------------------------------------------------------------
# Shared fixtures
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


@pytest.fixture(scope="module")
def benchmark_token_gen():
    return DeterministicTokenGenerator(salt="benchmark-salt")


# ---------------------------------------------------------------------------
# 1. Crypto benchmarks — the cost every masked value pays once
# ---------------------------------------------------------------------------
class TestCryptoBenchmarks:
    """AES-256-GCM key generation, encryption, and decryption cost."""

    def test_key_generation_speed(self, benchmark):
        benchmark(AESGCMCipher.generate_key)

    def test_encrypt_short_text(self, benchmark, benchmark_cipher):
        benchmark(benchmark_cipher.encrypt, "john@acme.com", b"EMAIL")

    def test_encrypt_medium_text(self, benchmark, benchmark_cipher):
        text = "A" * 1000
        benchmark(benchmark_cipher.encrypt, text, b"TEXT")

    def test_encrypt_decrypt_roundtrip(self, benchmark, benchmark_cipher):
        def roundtrip():
            enc = benchmark_cipher.encrypt("john@acme.com", b"EMAIL")
            benchmark_cipher.decrypt(enc.ciphertext, enc.iv, enc.tag, b"EMAIL")
        benchmark(roundtrip)


# ---------------------------------------------------------------------------
# 2. Token generator benchmarks — placeholder minting/parsing cost
# ---------------------------------------------------------------------------
class TestTokenBenchmarks:
    """DeterministicTokenGenerator: generate, parse, scan, and hash cost."""

    def test_generate_token(self, benchmark, benchmark_token_gen):
        benchmark(benchmark_token_gen.generate, "john@acme.com", EntityType.EMAIL)

    def test_parse_token(self, benchmark, benchmark_token_gen):
        token = benchmark_token_gen.generate("john@acme.com", EntityType.EMAIL)
        benchmark(benchmark_token_gen.parse_token, token)

    def test_find_tokens_in_text(self, benchmark, benchmark_token_gen):
        """Scanning a masked document for {{TYPE:xxxxx}} placeholders (unmask's first step)."""
        tokens = [
            benchmark_token_gen.generate(f"user{i}@acme.com", EntityType.EMAIL) for i in range(20)
        ]
        text = " ".join(f"Contact {t} for details." for t in tokens)
        benchmark(benchmark_token_gen.find_tokens_in_text, text)

    def test_compute_value_hash(self, benchmark, benchmark_token_gen):
        benchmark(benchmark_token_gen.compute_value_hash, "john@acme.com")


# ---------------------------------------------------------------------------
# 3. NER detection benchmarks — cost of finding PII before it's masked
# ---------------------------------------------------------------------------
class TestNERBenchmarks:
    """Regex-layer detection cost across text shapes (sparse, dense, large)."""

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
# 4. Storage backend benchmarks — the actual comparison table
# ---------------------------------------------------------------------------
# put() and get() are benchmarked identically across all four StorageBackend
# implementations using the exact same TokenRecord shape (see make_record),
# so the numbers in docs/BENCHMARKS.md are a fair apples-to-apples
# comparison, not an artifact of different payload sizes.
#
# put() benchmarks use benchmark.pedantic() with a fixed, modest round
# count instead of pytest-benchmark's default auto-calibration: put() is
# stateful (each round must write a *new* token, or Postgres/Redis's
# insert-if-absent semantics would silently turn every round after the
# first into a no-op) and, for FileSystemStorage, every put() does a full
# file rewrite — left uncapped, calibration alone could run thousands of
# disk writes. get() is a pure read with no side effects, so it's left on
# pytest-benchmark's normal auto-calibrated timing.


class TestStorageBackendBenchmarks:
    # -- InMemoryStorage: plain dict + asyncio.Lock, no I/O -----------------
    def test_memory_put(self, benchmark):
        storage = InMemoryStorage()
        counter = itertools.count()

        def do_put():
            n = next(counter)
            run_async(storage.put(make_record(f"{{{{EMAIL:mem{n:05x}}}}}", f"mem-hash-{n}")))

        benchmark.pedantic(do_put, rounds=2000, iterations=1)

    def test_memory_get(self, benchmark):
        storage = InMemoryStorage()
        run_async(storage.put(make_record("{{EMAIL:memget}}", "mem-hash-get")))
        benchmark(lambda: run_async(storage.get("{{EMAIL:memget}}")))

    # -- FileSystemStorage: JSON file, full rewrite per put() ---------------
    def test_filesystem_put(self, benchmark, tmp_path):
        storage = FileSystemStorage(str(tmp_path / "bench_vault_put.json"))
        run_async(storage.connect())
        counter = itertools.count()

        def do_put():
            n = next(counter)
            run_async(storage.put(make_record(f"{{{{EMAIL:fs{n:05x}}}}}", f"fs-hash-{n}")))

        benchmark.pedantic(do_put, rounds=200, iterations=1)
        run_async(storage.close())

    def test_filesystem_get(self, benchmark, tmp_path):
        storage = FileSystemStorage(str(tmp_path / "bench_vault_get.json"))
        run_async(storage.connect())
        run_async(storage.put(make_record("{{EMAIL:fsget}}", "fs-hash-get")))
        benchmark(lambda: run_async(storage.get("{{EMAIL:fsget}}")))
        run_async(storage.close())

    # -- PostgresStorage: real network round-trip per call ------------------
    @pytest.fixture
    def pg_bench_storage(self):
        try:
            import asyncpg
        except ImportError:
            pytest.skip("asyncpg not installed — pip install 'pii-protect[postgres]'")

        from pii_protect.storage.postgres import PostgresStorage
        import uuid

        schema = f"pii_bench_{uuid.uuid4().hex[:12]}"
        storage = PostgresStorage(POSTGRES_DSN, schema=schema)
        try:
            run_async(storage.connect())
        except Exception as exc:
            pytest.skip(f"PostgreSQL not reachable at {POSTGRES_DSN}: {exc}")

        yield storage

        run_async(storage.close())

        async def _cleanup():
            conn = await asyncpg.connect(POSTGRES_DSN)
            try:
                await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            finally:
                await conn.close()

        run_async(_cleanup())

    def test_postgres_put(self, benchmark, pg_bench_storage):
        counter = itertools.count()

        def do_put():
            n = next(counter)
            run_async(pg_bench_storage.put(make_record(f"{{{{EMAIL:pg{n:05x}}}}}", f"pg-hash-{n}")))

        benchmark.pedantic(do_put, rounds=50, iterations=1)

    def test_postgres_get(self, benchmark, pg_bench_storage):
        run_async(pg_bench_storage.put(make_record("{{EMAIL:pgget}}", "pg-hash-get")))
        benchmark(lambda: run_async(pg_bench_storage.get("{{EMAIL:pgget}}")))

    # -- RedisStorage: real network round-trip per call ----------------------
    @pytest.fixture
    def redis_bench_storage(self):
        try:
            from redis import asyncio as redis_asyncio  # noqa: F401
        except ImportError:
            pytest.skip("redis not installed — pip install 'pii-protect[redis]'")

        from pii_protect.storage.redis_backend import RedisStorage
        import uuid

        prefix = f"pii_bench:{uuid.uuid4().hex[:12]}:"
        storage = RedisStorage(url=REDIS_URL, key_prefix=prefix)
        try:
            run_async(storage.connect())
        except Exception as exc:
            pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")

        yield storage

        async def _cleanup():
            client = storage._client
            keys = [k async for k in client.scan_iter(match=f"{prefix}*")]
            if keys:
                await client.delete(*keys)

        run_async(_cleanup())
        run_async(storage.close())

    def test_redis_put(self, benchmark, redis_bench_storage):
        counter = itertools.count()

        def do_put():
            n = next(counter)
            run_async(redis_bench_storage.put(make_record(f"{{{{EMAIL:rd{n:05x}}}}}", f"rd-hash-{n}")))

        benchmark.pedantic(do_put, rounds=100, iterations=1)

    def test_redis_get(self, benchmark, redis_bench_storage):
        run_async(redis_bench_storage.put(make_record("{{EMAIL:rdget}}", "rd-hash-get")))
        benchmark(lambda: run_async(redis_bench_storage.get("{{EMAIL:rdget}}")))


# ---------------------------------------------------------------------------
# 5. End-to-end engine benchmarks — full mask()/unmask()/redact() cost
#    (NER + tokens + crypto + storage combined, over InMemoryStorage so the
#    numbers isolate the engine's own overhead from any one backend's cost;
#    see TestStorageBackendBenchmarks for backend-specific put()/get() cost)
#
# The engine is constructed and initialise()'d once per test, outside the
# benchmarked closure, so every number below is the operation's own cost —
# not engine-construction overhead — making mask/unmask/redact/mask_dict
# directly comparable to each other.
# ---------------------------------------------------------------------------
class TestEndToEndBenchmarks:
    @pytest.fixture
    def engine(self, benchmark_key):
        eng = PIIMaskingEngine(storage=InMemoryStorage(), encryption_key=benchmark_key)
        run_async(eng.initialise())
        yield eng
        run_async(eng.close())

    def test_mask_short_text(self, benchmark, engine):
        benchmark(
            lambda: run_async(engine.mask("Contact john@acme.com about GST 27AAPFU0939F1ZV"))
        )

    def test_unmask_short_text(self, benchmark, engine):
        """Isolates unmask() cost (find tokens, fetch, decrypt) from mask()'s."""
        masked = run_async(engine.mask("Contact john@acme.com about GST 27AAPFU0939F1ZV"))
        benchmark(lambda: run_async(engine.unmask(masked.masked_text)))

    def test_mask_unmask_roundtrip(self, benchmark, engine):
        async def roundtrip():
            result = await engine.mask("Contact john@acme.com about GST 27AAPFU0939F1ZV")
            await engine.unmask(result.masked_text)

        benchmark(lambda: run_async(roundtrip()))

    def test_redact_short_text(self, benchmark, engine):
        benchmark(engine.redact, "Contact john@acme.com about GST 27AAPFU0939F1ZV")

    def test_mask_dict_benchmark(self, benchmark, engine):
        data = {
            "email": "john@acme.com",
            "gst": "27AAPFU0939F1ZV",
            "phone": "+91 98765 43210",
            "pan": "ABCDE1234F",
        }
        benchmark(lambda: run_async(engine.mask_dict(data)))
