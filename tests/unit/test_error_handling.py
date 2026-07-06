"""Unit tests for error handling (pii_protect exceptions).

Tests cover:
- EngineNotInitialisedError when calling mask/unmask before init
- DecryptionError propagation
- StorageBackendError for backend failures
- OptionalDependencyMissingError for missing extras
- PIIShieldError as base class
- Error hierarchy and inheritance
"""
import pytest
from pii_protect import (
    PIIShieldError,
    EngineNotInitialisedError,
    DecryptionError,
    StorageBackendError,
    OptionalDependencyMissingError,
    PIIMaskingEngine,
)
from pii_protect.storage import InMemoryStorage


pytestmark = pytest.mark.unit


class TestErrorHierarchy:
    def test_all_errors_inherit_from_base(self):
        assert issubclass(EngineNotInitialisedError, PIIShieldError)
        assert issubclass(DecryptionError, PIIShieldError)
        assert issubclass(StorageBackendError, PIIShieldError)
        assert issubclass(OptionalDependencyMissingError, PIIShieldError)

    def test_base_error_is_exception(self):
        assert issubclass(PIIShieldError, Exception)

    def test_errors_can_be_raised_and_caught(self):
        with pytest.raises(PIIShieldError):
            raise EngineNotInitialisedError("test")
        with pytest.raises(PIIShieldError):
            raise DecryptionError("test")
        with pytest.raises(PIIShieldError):
            raise StorageBackendError("test")
        with pytest.raises(PIIShieldError):
            raise OptionalDependencyMissingError("feature", "extra", "package")


class TestEngineNotInitialised:
    @pytest.mark.asyncio
    async def test_mask_before_init_raises(self):
        engine = PIIMaskingEngine(storage=InMemoryStorage())
        with pytest.raises(EngineNotInitialisedError):
            await engine.mask("john@acme.com")

    @pytest.mark.asyncio
    async def test_unmask_before_init_raises(self):
        engine = PIIMaskingEngine(storage=InMemoryStorage())
        with pytest.raises(EngineNotInitialisedError):
            await engine.unmask("{{EMAIL:abc12}}")


class TestOptionalDependencyMissing:
    def test_error_message_is_descriptive(self):
        err = OptionalDependencyMissingError("spaCy NER", "spacy", "spacy")
        assert "spacy" in str(err).lower()
        assert "spacy ner" in str(err).lower()


class TestDecryptionError:
    def test_wrong_key_decryption(self):
        from pii_protect.crypto import AESGCMCipher
        c1 = AESGCMCipher(AESGCMCipher.generate_key())
        c2 = AESGCMCipher(AESGCMCipher.generate_key())
        encrypted = c1.encrypt("test", b"EMAIL")
        with pytest.raises(DecryptionError):
            c2.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, b"EMAIL")
