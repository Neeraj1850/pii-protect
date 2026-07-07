# pii-protect Performance Benchmarks

## Basic Comparison

`pii-protect` ships four interchangeable `StorageBackend` implementations behind one interface (`put`, `get`, `get_many`, `find_by_value_hash`, `touch`). Which one you pick changes latency, persistence, and operational overhead, not correctness — the same `PIIMaskingEngine.mask()/unmask()` calls work unmodified against any of them.

| Category | InMemoryStorage | FileSystemStorage | PostgresStorage | RedisStorage |
|---|---|---|---|---|
| Persistence | None — lost on process exit | Single JSON file on disk | Real database, durable | In-memory server, durable with AOF/RDB config |
| Process model | In-process only | Single process (in-process lock, no cross-process coordination) | Multi-process / multi-host safe | Multi-process / multi-host safe |
| Network hop | None | None (disk I/O) | Yes — TCP round trip per call | Yes — TCP round trip per call |
| Write cost per `put()` | O(1) dict insert | Full-vault JSON rewrite (write-to-temp + `os.replace`) | Single-row `INSERT ... ON CONFLICT DO NOTHING` | `HSET` + optional TTL `EXPIRE` |
| Audit logging | No | No | Yes — `access_log` table (`log_access()` override) | No |
| TTL / auto-expiry | No | No | No | Yes — `ttl_seconds` on the backend |
| Setup complexity | None | None | Schema auto-created on `connect()`, needs a running Postgres | Needs a running Redis |
| Best fit | Tests, short-lived scripts | Single-process CLIs, small on-prem deployments | Multi-instance production, compliance/audit requirements | Multi-instance production, lowest network latency, ephemeral vault data |

---

## Comparison using Code

The same benchmark harness (`tests/benchmarks/test_performance.py`, `pytest-benchmark`) drives every backend through the identical `TokenRecord` shape, so the numbers below are apples-to-apples — not an artifact of one backend getting an easier payload:

```python
def make_record(token: str, value_hash: str) -> TokenRecord:
    return TokenRecord(
        token_value=token, entity_type="EMAIL",
        ciphertext=b"0123456789abcdef", iv=b"iv_12_bytes_",
        tag=b"tag_16_bytes____", original_length=13, value_hash=value_hash,
    )
```

`put()` is stateful — writing the same token twice is a no-op on every backend (idempotent by design), so a naive benchmark that replays one record thousands of times would just be timing "no-op" after round 1. Each `put()` benchmark instead mints a fresh token per round and runs a **fixed, modest round count** via `benchmark.pedantic()` instead of `pytest-benchmark`'s free-running calibration:

```python
def test_postgres_put(self, benchmark, pg_bench_storage):
    counter = itertools.count()

    def do_put():
        n = next(counter)
        run_async(pg_bench_storage.put(make_record(f"{{{{EMAIL:pg{n:05x}}}}}", f"pg-hash-{n}")))

    benchmark.pedantic(do_put, rounds=50, iterations=1)
```

`get()` is a pure read with no side effects, so it's left on `pytest-benchmark`'s normal auto-calibrated timing (it decides how many rounds fit in ~1 second).

Run it yourself: `pytest tests/benchmarks/ --benchmark-only`.

---

## Benchmark Results

Machine: macOS 26.5.1, Apple Silicon (arm64), Python 3.13.12. PostgreSQL 18.3 and Redis 8.8.0 running **locally** (loopback, no real network hop) — over an actual network, Postgres/Redis numbers will be higher; Memory/FileSystem numbers won't change.

### Storage backends — `put()` / `get()`

| Backend | put() mean | put() ops/sec | get() mean | get() ops/sec |
|---|---:|---:|---:|---:|
| InMemoryStorage | 33.94 µs | 29,465 | 31.67 µs | 31,573 |
| FileSystemStorage | 744.00 µs | 1,344 | 31.01 µs | 32,252 |
| PostgresStorage | 390.19 µs | 2,563 | 186.67 µs | 5,357 |
| RedisStorage | 317.95 µs | 3,145 | 122.49 µs | 8,164 |

### Crypto (AES-256-GCM)

| Operation | Mean |
|---|---:|
| Key generation | 0.92 µs |
| Encrypt (short value, e.g. an email) | 1.74 µs |
| Encrypt (1 KB value) | 1.88 µs |
| Encrypt + decrypt roundtrip | 2.31 µs |

### Tokens (`DeterministicTokenGenerator`)

| Operation | Mean |
|---|---:|
| Generate token | 0.59 µs |
| Parse token | 0.34 µs |
| Scan text for 20 tokens | 2.82 µs |
| Compute value hash | 0.33 µs |

### NER detection (regex layer)

| Input | Mean |
|---|---:|
| Short text, 1 email + 1 GST | 8.21 µs |
| No PII (plain sentence) | 6.19 µs |
| Dense (5 PII types in one line) | 17.51 µs |
| Large (~5 KB, 10 paragraphs) | 774.81 µs |

### End-to-end engine (over InMemoryStorage)

| Operation | Mean | Ops/sec |
|---|---:|---:|
| `redact()` | 8.80 µs | 113,574 |
| `unmask()` | 36.62 µs | 27,310 |
| `mask()` | 44.68 µs | 22,382 |
| `mask()` + `unmask()` roundtrip | 48.27 µs | 20,715 |
| `mask_dict()` (4 fields) | 56.76 µs | 17,619 |

---

## Breaking Down Each Metric

### 1. Storage protocol — why `put()` costs so much more than `get()`

This is the biggest structural difference and explains most of the table.

- **InMemoryStorage**: `put()` and `get()` are both a plain dict write/read behind an `asyncio.Lock` — no I/O either way, so they cost about the same (~32-34 µs).
- **FileSystemStorage**: `put()` serialises the *entire vault* to JSON and rewrites the file on every single call (write-to-temp + `os.replace`, for crash safety). `get()` is just a dict lookup in memory (the file is loaded once on `connect()`). That's why FileSystemStorage's `put()` (744 µs) is **24x** its own `get()` (31 µs) — and the slowest `put()` of all four backends, slower even than a real network round trip to Postgres.
- **PostgresStorage / RedisStorage**: `put()` and `get()` both pay a TCP round trip, so they're much closer to each other than FileSystemStorage's are. Postgres's `put()` is an `INSERT` (write path, WAL-durable) vs. an indexed `SELECT` for `get()` — write costs more than read, as expected. Redis's `put()` writes a hash plus a secondary dedup-index key (two Redis commands) vs. one `HGETALL` for `get()` — same shape, smaller gap because Redis has no durability guarantee to pay for by default.

```
InMemoryStorage:    Code → dict[key] = value                    (no I/O)
FileSystemStorage:  Code → serialize ALL records → write file   (O(vault size) per put!)
PostgresStorage:     Code ←→ TCP ←→ Postgres (INSERT/SELECT, WAL)
RedisStorage:        Code ←→ TCP ←→ Redis (HSET+SET / HGETALL)
```

**Why it matters for choosing a backend:** FileSystemStorage's `put()` cost is *O(vault size)*, not O(1) — every write re-serialises every record ever stored. It's fine for a small on-prem vault (hundreds to low thousands of tokens), but it will keep getting slower as the vault grows, unlike the other three backends whose `put()` cost stays flat regardless of how much has been stored before.

### 2. Postgres beats Redis on `get()` for the wrong reason to rely on

Redis's `get()` (122 µs) is nearly **34% faster** than Postgres's (187 µs) — expected, since Redis is an in-memory key-value store built for exactly this access pattern, while Postgres pays for MVCC/index lookup overhead per query. But both numbers here are measured over loopback with zero network latency; over a real network, TCP round-trip time will dominate both equally and this gap will shrink in relative terms. Don't read "Redis get() is 34% faster" as "Redis get() is 34% faster in production" — re-run this benchmark against your actual deployment topology before making a capacity-planning decision on it.

### 3. The in-memory backend is not free — it has a ~30 µs floor

InMemoryStorage's `put()`/`get()` (~32-34 µs) look surprisingly close to FileSystemStorage's `get()` (31 µs) and nowhere near as cheap as, say, `compute_value_hash()` (0.33 µs) — despite InMemoryStorage doing nothing but a dict operation. The gap is `asyncio` dispatch overhead (an `async def` call, an `await`, and an `asyncio.Lock` acquire/release), not the dict itself. Every `StorageBackend` method is declared `async` — including `InMemoryStorage`'s, which has no actual I/O to await — so that overhead is a fixed cost of the pluggable-backend design, paid even by the fastest possible implementation.

### 4. Crypto and token generation are effectively free next to storage

AES-256-GCM encryption (1.7-2.3 µs) and token generation (0.3-0.6 µs) are both **one to two orders of magnitude cheaper** than any storage `put()`. Doubling your text size (short vs. 1 KB encrypt: 1.74 µs → 1.88 µs, +8%) barely moves the needle. If you're optimizing a slow `mask()` call, the storage backend is always where the time is going — not the cipher, not the token math.

### 5. NER detection scales with text length, not PII density

Dense text (5 PII types packed into one line) costs 17.51 µs — barely more than the short-text case (8.21 µs) despite detecting far more entities. The large-text case (~5 KB across 10 paragraphs, similar PII density to the dense case) costs 774.81 µs — a **44x** jump driven by *total characters scanned*, not entity count. Every regex pattern in `RegexPatternLibrary` runs against the full input on every `detect()` call; call it once per document, not once per paragraph, if you're chunking large inputs.

### 6. `mask()` costs roughly one `detect()` plus one storage `put()`

`redact()` (8.80 µs) is essentially the NER detection cost alone (comparable to the 8.21 µs short-text detection benchmark) plus a string replace — it never touches storage or the cipher, which is the whole point of `redact()` being irreversible. `mask()` (44.68 µs) is close to `redact()`'s detection cost *plus* InMemoryStorage's `put()` cost (8.80 + 33.94 ≈ 42.74 µs, in the right ballpark alongside the AES encrypt and dedup-lookup calls also in the critical path). Swap InMemoryStorage for FileSystemStorage in a `mask()`-heavy workload and expect the difference to track FileSystemStorage's much higher `put()` cost almost directly.

### 7. `unmask()` is cheaper than `mask()`, and the roundtrip is cheaper than the sum of its parts

`unmask()` (36.62 µs) beats `mask()` (44.68 µs) because it skips the dedup lookup (`find_by_value_hash`) that every `mask()` call pays to decide whether a value has already been tokenised in this scope — `unmask()` only needs `get_many()` by token value. The roundtrip benchmark (`mask()` then `unmask()` the same text, 48.27 µs) is *not* simply `44.68 + 36.62 = 81.30 µs`: the second `mask()` call inside the roundtrip test is a repeat of the same text, so its dedup lookup finds the already-stored token and skips the `put()` and encrypt steps entirely — a real illustration of `pii-protect`'s within-scope deduplication paying off, not a benchmarking artifact.

### 8. This report sits on top of a fully-tested implementation, not just raw numbers

Every backend measured here (`InMemoryStorage`, `FileSystemStorage`, `PostgresStorage`, `RedisStorage`) has full correctness coverage in `tests/unit/storage/` and `tests/integration/test_{postgres,redis}_engine.py` — put/get roundtrips, idempotency, scope-based deduplication, TTL expiry, audit logging, and full `mask()`/`unmask()` flows against real running Postgres/Redis instances. These benchmarks measure *how fast* correct behaviour is, not a stand-in for testing that the behaviour is correct in the first place.

---

## Final Summary

For local development and the unit/integration test suite itself, **InMemoryStorage** is the right default — zero setup, fastest `put()`, and isolation between test runs comes for free. For a single-process CLI or a small on-prem deployment that shouldn't take a database dependency, **FileSystemStorage** works well up to a modest vault size, but its per-`put()` full-file rewrite means it will not scale gracefully into the tens of thousands of tokens. For anything running as more than one process or needing a durable, shared, audited vault — which is the normal shape of a production PII-masking deployment — the choice is between **PostgresStorage**, which is the only backend with built-in audit logging (`access_log`) and the strongest durability guarantees, and **RedisStorage**, which is faster on both `put()` and `get()` and adds TTL-based auto-expiry, at the cost of no audit trail and Redis's weaker default durability model. Pick Postgres when you need to prove who accessed what and when; pick Redis when you need the lowest latency and the vault entries are allowed to expire.
