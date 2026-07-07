"""Unit tests for DeterministicTokenGenerator (pii_protect.tokens).

Tests cover:
- Token format ({{TYPE:xxxxx}}, lowercase hex suffix)
- Determinism: same value + entity_type + salt -> same token
- Different salts / values / entity types -> different tokens
- parse_token() roundtrip and rejection of malformed tokens
- find_tokens_in_text() locates all placeholders with correct offsets
- compute_value_hash() is salt-independent and stable
"""
import re

import pytest

from pii_protect.tokens import DeterministicTokenGenerator, TOKEN_PATTERN
from pii_protect.types import EntityType


pytestmark = pytest.mark.unit


class TestTokenFormat:
    @pytest.fixture
    def gen(self):
        return DeterministicTokenGenerator(salt="test-salt")

    def test_token_matches_pattern(self, gen):
        token = gen.generate("john@acme.com", EntityType.EMAIL)
        assert TOKEN_PATTERN.fullmatch(token)

    def test_token_starts_with_entity_label(self, gen):
        token = gen.generate("john@acme.com", EntityType.EMAIL)
        assert token.startswith("{{EMAIL:")

    def test_suffix_is_lowercase_hex_5_chars(self, gen):
        token = gen.generate("john@acme.com", EntityType.EMAIL)
        suffix = token.split(":")[1].rstrip("}")
        assert len(suffix) == 5
        assert re.fullmatch(r"[0-9a-f]{5}", suffix)


class TestDeterminism:
    def test_same_value_same_salt_same_token(self):
        gen = DeterministicTokenGenerator(salt="fixed-salt")
        t1 = gen.generate("john@acme.com", EntityType.EMAIL)
        t2 = gen.generate("john@acme.com", EntityType.EMAIL)
        assert t1 == t2

    def test_different_value_different_token(self):
        gen = DeterministicTokenGenerator(salt="fixed-salt")
        t1 = gen.generate("john@acme.com", EntityType.EMAIL)
        t2 = gen.generate("jane@acme.com", EntityType.EMAIL)
        assert t1 != t2

    def test_different_entity_type_different_token(self):
        gen = DeterministicTokenGenerator(salt="fixed-salt")
        t1 = gen.generate("27AAPFU0939F1ZV", EntityType.GST)
        t2 = gen.generate("27AAPFU0939F1ZV", EntityType.PAN)
        assert t1 != t2

    def test_different_salt_different_token(self):
        gen_a = DeterministicTokenGenerator(salt="salt-a")
        gen_b = DeterministicTokenGenerator(salt="salt-b")
        t1 = gen_a.generate("john@acme.com", EntityType.EMAIL)
        t2 = gen_b.generate("john@acme.com", EntityType.EMAIL)
        assert t1 != t2

    def test_env_salt_used_when_no_explicit_salt(self, monkeypatch):
        monkeypatch.setenv("PII_SHIELD_SALT", "env-salt-value")
        gen_env = DeterministicTokenGenerator()
        gen_explicit = DeterministicTokenGenerator(salt="env-salt-value")
        assert gen_env.generate("x@y.com", EntityType.EMAIL) == gen_explicit.generate(
            "x@y.com", EntityType.EMAIL
        )


class TestParseToken:
    @pytest.fixture
    def gen(self):
        return DeterministicTokenGenerator(salt="test-salt")

    def test_parse_valid_token_roundtrip(self, gen):
        token = gen.generate("john@acme.com", EntityType.EMAIL)
        parsed = gen.parse_token(token)
        assert parsed is not None
        entity_type, suffix = parsed
        assert entity_type == EntityType.EMAIL
        assert token == f"{{{{{entity_type.value}:{suffix}}}}}"

    def test_parse_malformed_token_returns_none(self, gen):
        assert gen.parse_token("not-a-token") is None
        assert gen.parse_token("{{EMAIL:xyz}}") is None  # suffix not 5 hex chars
        assert gen.parse_token("{{email:abcde}}") is None  # lowercase label
        assert gen.parse_token("") is None

    def test_parse_unknown_entity_type_returns_none(self, gen):
        assert gen.parse_token("{{NOT_A_REAL_TYPE:abcde}}") is None


class TestFindTokensInText:
    @pytest.fixture
    def gen(self):
        return DeterministicTokenGenerator(salt="test-salt")

    def test_finds_single_token(self, gen):
        token = gen.generate("john@acme.com", EntityType.EMAIL)
        text = f"Contact {token} for details."
        positions = gen.find_tokens_in_text(text)
        assert len(positions) == 1
        start, end, found = positions[0]
        assert found == token
        assert text[start:end] == token

    def test_finds_multiple_tokens_in_order(self, gen):
        t1 = gen.generate("john@acme.com", EntityType.EMAIL)
        t2 = gen.generate("27AAPFU0939F1ZV", EntityType.GST)
        text = f"Email {t1}, GST {t2}"
        positions = gen.find_tokens_in_text(text)
        assert len(positions) == 2
        assert positions[0][2] == t1
        assert positions[1][2] == t2

    def test_no_tokens_returns_empty_list(self, gen):
        assert gen.find_tokens_in_text("No placeholders here.") == []

    def test_ignores_malformed_lookalike(self, gen):
        assert gen.find_tokens_in_text("{{EMAIL:zz}} {{email:abcde}}") == []


class TestComputeValueHash:
    def test_same_value_same_hash_regardless_of_salt(self):
        gen_a = DeterministicTokenGenerator(salt="salt-a")
        gen_b = DeterministicTokenGenerator(salt="salt-b")
        assert gen_a.compute_value_hash("john@acme.com") == gen_b.compute_value_hash(
            "john@acme.com"
        )

    def test_different_value_different_hash(self):
        gen = DeterministicTokenGenerator(salt="test-salt")
        assert gen.compute_value_hash("john@acme.com") != gen.compute_value_hash(
            "jane@acme.com"
        )

    def test_hash_is_64_char_hex(self):
        gen = DeterministicTokenGenerator(salt="test-salt")
        h = gen.compute_value_hash("john@acme.com")
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h)
