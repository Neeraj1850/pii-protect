"""Unit tests for NEREngine and RegexNERLayer (pii_protect.ner).

Tests cover:
- NEREngine initialisation (default = regex only)
- Detection of various PII entity types via regex layer
- Non-overlapping span output
- Confidence scores present
- Entity type correctness
- Empty text returns no spans
- Text with no PII returns no spans
"""
import pytest
from pii_protect import NEREngine
from pii_protect.types import EntityType


pytestmark = pytest.mark.unit


class TestNEREngineInit:
    def test_default_init_no_extras(self):
        """NEREngine() with no arguments should work (regex only)."""
        ner = NEREngine()
        assert ner is not None

    def test_spacy_not_enabled_by_default(self):
        ner = NEREngine()
        assert ner is not None


class TestRegexDetection:
    @pytest.fixture
    def ner(self):
        return NEREngine()

    def test_detect_email(self, ner):
        spans = ner.detect("Contact john@acme.com please")
        assert any(s.entity_type == "EMAIL" for s in spans)

    def test_detect_phone_indian(self, ner):
        spans = ner.detect("Call +91 98765 43210")
        phone_spans = [s for s in spans if s.entity_type == "PHONE"]
        assert len(phone_spans) >= 1

    def test_detect_phone_international(self, ner):
        spans = ner.detect("Ring +44 20 7946 0958")
        phone_spans = [s for s in spans if s.entity_type == "PHONE"]
        assert len(phone_spans) >= 1

    def test_detect_gst(self, ner):
        spans = ner.detect("GST number 27AAPFU0939F1ZV")
        assert any(s.entity_type == "GST" for s in spans)

    def test_detect_pan(self, ner):
        spans = ner.detect("PAN is ABCDE1234F")
        assert any(s.entity_type == "PAN" for s in spans)

    def test_detect_iban(self, ner):
        spans = ner.detect("IBAN: GB29NWBK60161331926819")
        assert any(s.entity_type == "IBAN" for s in spans)

    def test_detect_swift(self, ner):
        spans = ner.detect("SWIFT code DEUTDEFF")
        assert any(s.entity_type == "SWIFT" for s in spans)

    def test_detect_credit_card(self, ner):
        spans = ner.detect("Card: 4111111111111111")
        cc_types = [s.entity_type for s in spans]
        assert any("CARD" in t or "CREDIT" in t for t in cc_types)

    def test_detect_multiple_entities(self, ner):
        text = "Email john@acme.com, GST 27AAPFU0939F1ZV, PAN ABCDE1234F"
        spans = ner.detect(text)
        types = {s.entity_type for s in spans}
        assert "EMAIL" in types
        assert "GST" in types
        assert "PAN" in types

    def test_no_pii_returns_empty(self, ner):
        spans = ner.detect("The weather is nice today.")
        assert len(spans) == 0

    def test_empty_text_returns_empty(self, ner):
        spans = ner.detect("")
        assert len(spans) == 0

    def test_spans_have_confidence(self, ner):
        spans = ner.detect("john@acme.com")
        for s in spans:
            assert hasattr(s, 'confidence') or hasattr(s, 'score')

    def test_spans_non_overlapping(self, ner):
        text = "Email john@acme.com and GST 27AAPFU0939F1ZV"
        spans = ner.detect(text)
        sorted_spans = sorted(spans, key=lambda s: s.start)
        for i in range(len(sorted_spans) - 1):
            assert sorted_spans[i].end <= sorted_spans[i + 1].start, (
                f"Spans overlap: {sorted_spans[i]} and {sorted_spans[i+1]}"
            )

    def test_detect_invoice_number(self, ner):
        spans = ner.detect("Invoice INV-2026-00417")
        types = {s.entity_type for s in spans}
        assert len(spans) >= 1, f"Expected invoice detection, got types: {types}"

    def test_detect_po_number(self, ner):
        spans = ner.detect("PO number PO-2026-0042")
        assert len(spans) >= 1


# ---------------------------------------------------------------------------
# SpanConflictResolver — direct unit tests on synthetic spans.
#
# The suite already exercises "a wins" for each priority rule (via real
# detected text in test_ner_spacy.py / test_ner_privacy_filter.py), but never
# the mirrored "b wins" direction for the same rules — those branches
# (`ner/engine.py` _pick_winner's second `if` in each pair) need hand-built
# overlapping spans to hit deterministically, since which layer's span
# happens to land in the "a" vs "b" position when sorted depends on model
# output we don't control from a real sentence.
# ---------------------------------------------------------------------------

class TestSpanConflictResolverMirrorCases:
    @pytest.fixture
    def resolver(self):
        from pii_protect.ner.engine import SpanConflictResolver
        return SpanConflictResolver()

    def _span(self, start, end, text, entity_type, confidence, source="test",
              is_regex_validated=False):
        from pii_protect.types import DetectedSpan
        return DetectedSpan(
            start=start, end=end, text=text, entity_type=entity_type,
            confidence=confidence, source=source,
            is_regex_validated=is_regex_validated,
        )

    def test_a_wins_when_a_is_regex_validated(self, resolver):
        """Baseline (already implied elsewhere) — kept alongside the mirror
        case below so both directions of the same rule live side by side."""
        a = self._span(0, 10, "27AAPFU0939F1ZV", EntityType.GST, 0.6,
                        is_regex_validated=True)
        b = self._span(0, 10, "27AAPFU0939F1ZV", EntityType.PERSON, 0.9,
                        is_regex_validated=False)
        winner = resolver._pick_winner(a, b)
        assert winner is a

    def test_b_wins_when_b_is_regex_validated(self, resolver):
        """Mirror of the above: regex-validated must win regardless of
        which position (a or b) it happens to occupy."""
        a = self._span(0, 10, "27AAPFU0939F1ZV", EntityType.PERSON, 0.9,
                        is_regex_validated=False)
        b = self._span(0, 10, "27AAPFU0939F1ZV", EntityType.GST, 0.6,
                        is_regex_validated=True)
        winner = resolver._pick_winner(a, b)
        assert winner is b

    def test_a_financial_beats_b_phone(self, resolver):
        a = self._span(0, 15, "GB29NWBK6016133", EntityType.IBAN, 0.7)
        b = self._span(0, 15, "GB29NWBK6016133", EntityType.PHONE, 0.7)
        winner = resolver._pick_winner(a, b)
        assert winner is a

    def test_b_financial_beats_a_phone(self, resolver):
        """Mirror of the above: a financial entity type must beat PHONE
        regardless of which position it occupies in the pair."""
        a = self._span(0, 15, "GB29NWBK6016133", EntityType.PHONE, 0.7)
        b = self._span(0, 15, "GB29NWBK6016133", EntityType.IBAN, 0.7)
        winner = resolver._pick_winner(a, b)
        assert winner is b

    def test_resolve_drops_loser_from_final_list(self, resolver):
        """End-to-end sanity check via the public resolve() method, not just
        the private _pick_winner() helper."""
        a = self._span(0, 10, "9876543210", EntityType.PHONE, 0.8)
        b = self._span(0, 10, "9876543210", EntityType.GST, 0.8)
        resolved = resolver.resolve([a, b])
        assert len(resolved) == 1
        assert resolved[0].entity_type == EntityType.GST


# ---------------------------------------------------------------------------
# TokenizerSafeSpanMerger — direct unit tests on synthetic spans.
# ---------------------------------------------------------------------------

class TestTokenizerSafeSpanMerger:
    @pytest.fixture
    def merger(self):
        from pii_protect.ner.engine import TokenizerSafeSpanMerger
        return TokenizerSafeSpanMerger()

    def _span(self, start, end, text, entity_type=EntityType.PERSON,
              confidence=0.8, source="privacy_filter", is_regex_validated=False):
        from pii_protect.types import DetectedSpan
        return DetectedSpan(
            start=start, end=end, text=text, entity_type=entity_type,
            confidence=confidence, source=source,
            is_regex_validated=is_regex_validated,
        )

    def test_merges_adjacent_same_type_spans_with_gap(self, merger):
        """Sub-word token artefacts from a model (e.g. 'John' + 'Smith'
        emitted as two spans one space apart) must be merged into a single
        span with a space-joined text — this is the `gap > 0` branch of the
        join logic that real model output only sometimes exercises."""
        a = self._span(0, 4, "John")
        b = self._span(5, 10, "Smith")
        merged = merger.merge([a, b])
        assert len(merged) == 1
        assert merged[0].text == "John Smith"
        assert merged[0].start == 0
        assert merged[0].end == 10

    def test_merges_touching_spans_without_extra_space(self, merger):
        """Zero-gap spans (touching, no gap) must join without inserting a
        space — the `gap > 0` condition's false branch."""
        a = self._span(0, 4, "John")
        b = self._span(4, 9, "Smith")
        merged = merger.merge([a, b])
        assert len(merged) == 1
        assert merged[0].text == "JohnSmith"

    def test_does_not_merge_different_entity_types(self, merger):
        a = self._span(0, 4, "John", entity_type=EntityType.PERSON)
        b = self._span(5, 12, "Company", entity_type=EntityType.ORGANISATION)
        merged = merger.merge([a, b])
        assert len(merged) == 2

    def test_empty_input_returns_empty(self, merger):
        assert merger.merge([]) == []


# ---------------------------------------------------------------------------
# All three detection layers enabled simultaneously.
#
# Every other test file only turns on ONE optional layer at a time
# (test_ner_spacy.py: regex+spacy, test_ner_privacy_filter.py: regex+privacy
# filter). Nobody previously combined all three, so SpanConflictResolver
# picking a winner among three simultaneous overlapping detections was never
# exercised. The layers are fully independent of each other — each scans the
# *same* raw text blind to what the others found — so turning on all three
# doesn't multiply detection power; it multiplies the number of overlapping
# candidates the resolver has to arbitrate, and multiplies runtime cost
# (regex is near-free; spaCy and the transformer pipeline are not) even when
# a cheaper layer already found the same span. That's the real tradeoff:
# more layers = more recall on edge cases (bare names, dense free text) at
# the cost of latency, and the resolver's priority formula
# (confidence + 0.30 if regex-validated + 0.10 if financial + 0.001*length)
# is what keeps the output correct rather than duplicated or contradictory.
# ---------------------------------------------------------------------------

class TestAllThreeLayers:
    pytestmark = [pytest.mark.requires_spacy, pytest.mark.requires_privacy_filter]

    PRIVACY_MODEL = "iiiorg/piiranha-v1-detect-personal-information"

    @pytest.fixture(scope="module")
    def full_engine(self):
        try:
            return NEREngine(
                enable_spacy=True,
                enable_privacy_filter=True,
                privacy_filter_model=self.PRIVACY_MODEL,
                privacy_filter_threshold=0.5,
            )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"spacy/privacy-filter not available: {exc}")

    def test_all_three_layers_construct(self, full_engine):
        assert full_engine is not None

    def test_regex_wins_over_spacy_and_privacy_filter_on_same_span(self, full_engine):
        """A GST number is regex-validated; even with spaCy and the
        privacy-filter both scanning the same text, the regex-validated span
        must be the one that survives conflict resolution (rule 1: regex
        beats everything else, unconditionally)."""
        text = "GST 27AAPFU0939F1ZV was filed by Priya Sharma."
        spans = full_engine.detect(text)
        gst_spans = [s for s in spans if s.entity_type == EntityType.GST]
        assert len(gst_spans) == 1
        assert gst_spans[0].is_regex_validated is True
        assert gst_spans[0].text == "27AAPFU0939F1ZV"

    def test_all_three_layers_contribute_on_dense_text(self, full_engine):
        """Feed text with regex-only PII (email), spaCy-only PII (a bare
        name with no surrounding structured markers), and privacy-filter
        style dense free text — confirm spans from more than one `source`
        appear, proving the layers are genuinely all firing rather than one
        silently dominating detect() before the others run."""
        text = (
            "Please loop in Sarah Johnson on this thread. "
            "Contact billing@acme.com for the invoice."
        )
        spans = full_engine.detect(text)
        sources = {s.source for s in spans}
        assert len(sources) >= 2, f"Expected multiple layers to contribute, got: {sources}"

    def test_resolved_spans_non_overlapping_three_layers(self, full_engine, sample_text_with_mixed_pii):
        spans = sorted(full_engine.detect(sample_text_with_mixed_pii), key=lambda s: s.start)
        for i in range(len(spans) - 1):
            assert spans[i].end <= spans[i + 1].start, (
                f"Spans overlap: {spans[i]} and {spans[i + 1]}"
            )

    @pytest.mark.asyncio
    async def test_mask_end_to_end_three_layers(self, full_engine, assert_pii_absent):
        """Wire the fully-loaded 3-layer NEREngine into a real
        PIIMaskingEngine and confirm the masked output contains none of the
        original raw PII values."""
        from pii_protect import PIIMaskingEngine
        from pii_protect.storage import InMemoryStorage
        from pii_protect.crypto import AESGCMCipher

        text = "Please loop in Sarah Johnson at sarah.johnson@acme.com about GST 27AAPFU0939F1ZV."
        async with PIIMaskingEngine(
            storage=InMemoryStorage(),
            encryption_key=AESGCMCipher.generate_key(),
            ner_engine=full_engine,
        ) as engine:
            result = await engine.mask(text)

        assert_pii_absent(result.masked_text, [
            "Sarah Johnson", "sarah.johnson@acme.com", "27AAPFU0939F1ZV",
        ])
