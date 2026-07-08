"""Unit tests for SpacyNERLayer and NEREngine(enable_spacy=True).

Requires the `spacy` extra plus the `en_core_web_sm` model
(`python -m spacy download en_core_web_sm`). This is the previously-untested
optional detection layer — the regex-only tests elsewhere in this suite
never exercise it. Tests cover:
- Standalone SpacyNERLayer.detect(): PERSON/ORG/GPE -> PERSON/ORGANISATION/ADDRESS
- Fixed confidence (0.75) and source="spacy" on every span
- NEREngine(enable_spacy=True) end-to-end wiring
- spaCy catching a bare name that the regex layer alone would miss
- Mixed-layer resolution: a regex-matched value and a spaCy-only name in the
  same document both surviving SpanConflictResolver
- Irrelevant spaCy entity types (e.g. DATE, CARDINAL) being dropped
- Graceful skip if spaCy or the model isn't installed
"""
import pytest

from pii_protect import NEREngine
from pii_protect.types import EntityType

pytestmark = [pytest.mark.unit, pytest.mark.requires_spacy]


@pytest.fixture(scope="module")
def spacy_layer():
    try:
        from pii_protect.ner.engine import SpacyNERLayer
        return SpacyNERLayer()
    except Exception as exc:  # noqa: BLE001 - missing spacy / model -> skip
        pytest.skip(f"spaCy or en_core_web_sm not available: {exc}")


@pytest.fixture(scope="module")
def spacy_engine():
    try:
        return NEREngine(enable_spacy=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"spaCy or en_core_web_sm not available: {exc}")


class TestSpacyNERLayerStandalone:
    def test_detects_person_name(self, spacy_layer):
        spans = spacy_layer.detect("John Smith joined the meeting yesterday.")
        persons = [s for s in spans if s.entity_type == EntityType.PERSON]
        assert len(persons) >= 1
        assert "John Smith" in persons[0].text

    def test_detects_organisation(self, spacy_layer):
        spans = spacy_layer.detect("She works at Acme Corporation.")
        orgs = [s for s in spans if s.entity_type == EntityType.ORGANISATION]
        assert len(orgs) >= 1

    def test_detects_location_as_address(self, spacy_layer):
        spans = spacy_layer.detect("The office is located in New York.")
        addresses = [s for s in spans if s.entity_type == EntityType.ADDRESS]
        assert len(addresses) >= 1

    def test_span_source_is_spacy(self, spacy_layer):
        spans = spacy_layer.detect("John Smith works here.")
        assert len(spans) >= 1
        for span in spans:
            assert span.source == "spacy"

    def test_span_confidence_is_fixed_075(self, spacy_layer):
        """SpacyNERLayer doesn't expose per-entity confidence, so it hardcodes 0.75."""
        spans = spacy_layer.detect("John Smith works here.")
        assert len(spans) >= 1
        for span in spans:
            assert span.confidence == 0.75

    def test_span_offsets_match_text(self, spacy_layer):
        text = "Contact John Smith about the project."
        spans = spacy_layer.detect(text)
        for span in spans:
            assert text[span.start:span.end] == span.text

    def test_irrelevant_entity_types_dropped(self, spacy_layer):
        """DATE/CARDINAL/etc (not in _SPACY_TO_ENTITY) must not appear as spans."""
        spans = spacy_layer.detect("The meeting is on January 5th with 12 attendees.")
        # None of DATE or CARDINAL map to a DetectedSpan entity_type we track
        for span in spans:
            assert span.entity_type in {
                EntityType.PERSON, EntityType.ORGANISATION, EntityType.ADDRESS
            }

    def test_no_entities_in_plain_text(self, spacy_layer):
        spans = spacy_layer.detect("The weather is nice today.")
        assert spans == []

    def test_empty_text(self, spacy_layer):
        spans = spacy_layer.detect("")
        assert spans == []


class TestNEREngineWithSpacy:
    def test_enable_spacy_true_constructs(self, spacy_engine):
        assert spacy_engine is not None

    def test_detects_bare_name_regex_would_miss(self, spacy_engine):
        """A name with no surrounding structured PII (no email/phone/GST) is
        invisible to the regex-only engine but should be caught with spaCy on."""
        text = "Please loop in Sarah Johnson on this thread."
        regex_only = NEREngine()
        regex_spans = regex_only.detect(text)
        assert not any(s.entity_type == EntityType.PERSON for s in regex_spans)

        spacy_spans = spacy_engine.detect(text)
        assert any(s.entity_type == EntityType.PERSON for s in spacy_spans)

    def test_mixed_regex_and_spacy_spans_both_survive(self, spacy_engine):
        """A regex-matched email and a spaCy-only name in one document: both
        must appear in the final conflict-resolved span list (non-overlapping,
        different types, so SpanConflictResolver shouldn't drop either)."""
        text = "Contact John Smith at john.smith@acme.com for details."
        spans = spacy_engine.detect(text)
        types = {s.entity_type for s in spans}
        assert EntityType.EMAIL in types
        assert EntityType.PERSON in types

    def test_resolved_spans_non_overlapping(self, spacy_engine):
        text = "Email Priya Sharma at priya.sharma@techcorp.in about the GST 27AAPFU0939F1ZV filing."
        spans = sorted(spacy_engine.detect(text), key=lambda s: s.start)
        for i in range(len(spans) - 1):
            assert spans[i].end <= spans[i + 1].start

    def test_mask_with_spacy_engine_masks_name(self, spacy_engine):
        """End-to-end: PIIMaskingEngine wired with a spaCy-enabled NEREngine
        should mask a bare name that the default regex-only engine wouldn't."""
        import asyncio
        from pii_protect import PIIMaskingEngine
        from pii_protect.storage import InMemoryStorage
        from pii_protect.crypto import AESGCMCipher

        async def run():
            async with PIIMaskingEngine(
                storage=InMemoryStorage(),
                encryption_key=AESGCMCipher.generate_key(),
                ner_engine=spacy_engine,
            ) as engine:
                result = await engine.mask("Please loop in Sarah Johnson on this thread.")
                assert "Sarah Johnson" not in result.masked_text
                assert "{{PERSON:" in result.masked_text

        asyncio.run(run())
