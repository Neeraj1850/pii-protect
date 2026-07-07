"""Unit tests for FileSystemStorage (pii_protect.storage).

Tests cover:
- put/get roundtrip with file persistence
- File creation on first put
- Multiple records stored and retrieved
- Persistence across instances (same file path)
"""
import pytest
import pytest_asyncio
from pathlib import Path
from pii_protect.storage import FileSystemStorage
from pii_protect.types import TokenRecord


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def make_record(token="{{EMAIL:fs001}}", value_hash="fshash1", scope=None):
    return TokenRecord(
        token_value=token,
        entity_type="EMAIL",
        ciphertext=b"encrypted_fs",
        iv=b"iv_fs_data",
        tag=b"tag_fs_data",
        original_length=13,
        value_hash=value_hash,
        scope=scope,
    )


class TestFileSystemStorage:
    @pytest.fixture
    def vault_path(self, tmp_path) -> str:
        return str(tmp_path / "vault.json")

    @pytest_asyncio.fixture
    async def storage(self, vault_path) -> FileSystemStorage:
        s = FileSystemStorage(vault_path)
        await s.connect()
        yield s
        await s.close()

    async def test_put_creates_file(self, storage, vault_path):
        await storage.put(make_record())
        assert Path(vault_path).exists()

    async def test_put_and_get_roundtrip(self, storage):
        record = make_record()
        await storage.put(record)
        result = await storage.get(record.token_value)
        assert result is not None
        assert result.token_value == record.token_value

    async def test_get_nonexistent(self, storage):
        result = await storage.get("{{FAKE:00000}}")
        assert result is None

    async def test_multiple_records(self, storage):
        r1 = make_record(token="{{EMAIL:fs010}}", value_hash="h10")
        r2 = make_record(token="{{PHONE:fs020}}", value_hash="h20")
        await storage.put(r1)
        await storage.put(r2)
        assert (await storage.get(r1.token_value)) is not None
        assert (await storage.get(r2.token_value)) is not None

    async def test_persistence_across_instances(self, vault_path):
        s1 = FileSystemStorage(vault_path)
        await s1.connect()
        record = make_record(token="{{EMAIL:pst01}}", value_hash="hpst")
        await s1.put(record)
        await s1.close()

        s2 = FileSystemStorage(vault_path)
        await s2.connect()
        result = await s2.get(record.token_value)
        assert result is not None
        assert result.token_value == record.token_value
        await s2.close()

    async def test_get_many(self, storage):
        r1 = make_record(token="{{A:fs030}}", value_hash="h30")
        r2 = make_record(token="{{B:fs040}}", value_hash="h40")
        await storage.put(r1)
        await storage.put(r2)
        results = await storage.get_many([r1.token_value, r2.token_value])
        assert len(results) == 2

    async def test_find_by_value_hash(self, storage):
        record = make_record(value_hash="fs_unique", scope="scope-1")
        await storage.put(record)
        found = await storage.find_by_value_hash("fs_unique", scope="scope-1")
        assert found == record.token_value
