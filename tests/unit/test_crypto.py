"""Unit tests for AESGCMCipher (pii_protect.crypto).

Tests cover:
- Key generation (valid 64-char hex)
- Encrypt/decrypt roundtrip
- Decryption with wrong key fails (DecryptionError)
- Decryption with tampered ciphertext fails
- AAD binding (encrypt with one entity type, decrypt with another fails)
- Fresh IV per encryption (same plaintext produces different ciphertext)
- Empty string encryption
- Unicode/multilingual text encryption
- Large text encryption
"""
import pytest
from pii_protect.crypto import AESGCMCipher
from pii_protect import DecryptionError


pytestmark = pytest.mark.unit


class TestKeyGeneration:
    def test_generate_key_returns_32_bytes(self):
        key = AESGCMCipher.generate_key()
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_generate_key_unique_each_call(self):
        keys = {AESGCMCipher.generate_key() for _ in range(50)}
        assert len(keys) == 50, "Key generation should produce unique keys"


class TestEncryptDecrypt:
    @pytest.fixture
    def cipher(self):
        return AESGCMCipher(AESGCMCipher.generate_key())

    def test_roundtrip_basic(self, cipher):
        plaintext = "john@acme.com"
        aad = b"EMAIL"
        encrypted = cipher.encrypt(plaintext, aad)
        decrypted = cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, aad)
        assert decrypted == plaintext

    def test_roundtrip_unicode(self, cipher):
        plaintext = "\u540d\u524d\u306f\u592a\u90ce\u3067\u3059"
        aad = b"PERSON"
        encrypted = cipher.encrypt(plaintext, aad)
        assert cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, aad) == plaintext

    def test_roundtrip_empty_string(self, cipher):
        aad = b"EMAIL"
        encrypted = cipher.encrypt("", aad)
        assert cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, aad) == ""

    def test_roundtrip_large_text(self, cipher):
        plaintext = "A" * 100_000
        aad = b"TEXT"
        encrypted = cipher.encrypt(plaintext, aad)
        assert cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, aad) == plaintext

    def test_fresh_iv_per_encrypt(self, cipher):
        """Same plaintext + AAD should produce different ciphertext each time."""
        aad = b"EMAIL"
        enc1 = cipher.encrypt("test@test.com", aad)
        enc2 = cipher.encrypt("test@test.com", aad)
        assert enc1.ciphertext != enc2.ciphertext, "Each encryption must use a fresh IV"
        assert enc1.iv != enc2.iv, "Each encryption must use a fresh IV"

    def test_decrypt_wrong_key_raises(self):
        cipher1 = AESGCMCipher(AESGCMCipher.generate_key())
        cipher2 = AESGCMCipher(AESGCMCipher.generate_key())
        aad = b"TYPE"
        encrypted = cipher1.encrypt("secret", aad)
        with pytest.raises(DecryptionError):
            cipher2.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, aad)

    def test_decrypt_wrong_aad_raises(self, cipher):
        """AAD is bound - decrypting with a different entity type must fail."""
        encrypted = cipher.encrypt("john@acme.com", b"EMAIL")
        with pytest.raises(DecryptionError):
            cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, b"PHONE")

    def test_decrypt_tampered_ciphertext_raises(self, cipher):
        aad = b"TYPE"
        encrypted = cipher.encrypt("sensitive", aad)
        # Tamper with the encrypted data
        tampered_ciphertext = encrypted.ciphertext[:-1] + bytes([encrypted.ciphertext[-1] ^ 0xFF])
        with pytest.raises((DecryptionError, Exception)):
            cipher.decrypt(tampered_ciphertext, encrypted.iv, encrypted.tag, aad)

    @pytest.mark.parametrize("plaintext,aad", [
        ("priya.sharma@techcorp.in", b"EMAIL"),
        ("+91 98765 43210", b"PHONE"),
        ("27AAPFU0939F1ZV", b"GST"),
        ("ABCDE1234F", b"PAN"),
        ("4111111111111111", b"CREDIT_CARD"),
        ("GB29NWBK60161331926819", b"IBAN"),
    ])
    def test_roundtrip_various_pii_types(self, cipher, plaintext, aad):
        encrypted = cipher.encrypt(plaintext, aad)
        assert cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, aad) == plaintext

    def test_init_wrong_key_length_raises(self):
        with pytest.raises(ValueError):
            AESGCMCipher(b"too-short-key")


class TestFromHex:
    """AESGCMCipher.from_hex() lets callers configure the engine with a
    64-character hex string instead of raw bytes (e.g. from an env var).
    Nothing previously exercised this alternate constructor."""

    def test_from_hex_roundtrip(self):
        cipher = AESGCMCipher.from_hex("00" * 32)
        encrypted = cipher.encrypt("secret@acme.com", b"EMAIL")
        decrypted = cipher.decrypt(encrypted.ciphertext, encrypted.iv, encrypted.tag, b"EMAIL")
        assert decrypted == "secret@acme.com"

    def test_from_hex_matches_equivalent_raw_key(self):
        raw_key = AESGCMCipher.generate_key()
        hex_key = raw_key.hex()
        cipher_from_hex = AESGCMCipher.from_hex(hex_key)
        cipher_from_bytes = AESGCMCipher(raw_key)
        assert cipher_from_hex.key == cipher_from_bytes.key

    def test_from_hex_strips_whitespace(self):
        cipher = AESGCMCipher.from_hex(f"  {'ab' * 32}  \n")
        assert len(cipher.key) == 32

    def test_from_hex_wrong_length_raises(self):
        with pytest.raises(ValueError):
            AESGCMCipher.from_hex("ab" * 10)

