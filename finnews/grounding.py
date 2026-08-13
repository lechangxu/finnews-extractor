"""Independent source-grounding verification for LLM extractions.

Extraction libraries (LangExtract, LLM structured output in general) *claim*
a span in the source for each extraction. This module does not trust the
claim: it re-verifies every extraction against the source text, recovers
spans the library failed to align, and flags what cannot be grounded —
per extraction, and per attribute value.

Zero dependencies, no API key, fully deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Verdict levels
GROUNDED = "grounded"      # text located in source + every attribute grounded
PARTIAL = "partial"        # text located, but ≥1 attribute value not found in source
UNGROUNDED = "ungrounded"  # extraction text itself not found in source → hallucination

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Remove all whitespace so line-wrapped source text still matches."""
    return _WS.sub("", s)


@dataclass
class Span:
    """Where the extraction text actually lives in the source."""
    start: int
    end: int
    method: str  # "interval_exact" | "exact_search" | "normalized_search"


@dataclass
class Extraction:
    extraction_class: str
    extraction_text: str
    attributes: dict = field(default_factory=dict)
    char_start: int | None = None   # library-claimed interval, if any
    char_end: int | None = None


@dataclass
class AttributeVerdict:
    name: str
    value: str
    grounded: bool


@dataclass
class ExtractionVerdict:
    extraction: Extraction
    verdict: str
    span: Span | None
    recovered: bool                 # library gave no/bad interval, we found it anyway
    attributes: list[AttributeVerdict] = field(default_factory=list)


def locate(source: str, text: str, char_start: int | None = None,
           char_end: int | None = None) -> Span | None:
    """Find where `text` lives in `source`, independent of any claimed interval.

    Order of attempts: trust-but-verify the claimed interval first (cheapest
    and it validates the library), then exact substring, then
    whitespace-normalized substring mapped back to original offsets.
    """
    if not text:
        return None
    if char_start is not None and char_end is not None:
        if 0 <= char_start < char_end <= len(source) \
                and source[char_start:char_end] == text:
            return Span(char_start, char_end, "interval_exact")

    idx = source.find(text)
    if idx >= 0:
        return Span(idx, idx + len(text), "exact_search")

    # Normalized path: strip whitespace from both sides, map the hit back.
    norm_source = _normalize(source)
    norm_text = _normalize(text)
    if not norm_text:
        return None
    n_idx = norm_source.find(norm_text)
    if n_idx < 0:
        return None
    # positions in `source` of its non-whitespace chars
    orig_positions = [i for i, ch in enumerate(source) if not ch.isspace()]
    start = orig_positions[n_idx]
    last_norm = n_idx + len(norm_text) - 1
    end = orig_positions[last_norm] + 1
    return Span(start, end, "normalized_search")


def verify_extraction(source: str, ext: Extraction) -> ExtractionVerdict:
    span = locate(source, ext.extraction_text, ext.char_start, ext.char_end)
    if span is None:
        return ExtractionVerdict(ext, UNGROUNDED, None, recovered=False)

    recovered = span.method != "interval_exact"
    attr_verdicts = []
    for k, v in (ext.attributes or {}).items():
        v = str(v)
        ok = (v in source) or (_normalize(v) in _normalize(source))
        attr_verdicts.append(AttributeVerdict(k, v, ok))
    all_ok = all(a.grounded for a in attr_verdicts)
    return ExtractionVerdict(
        ext, GROUNDED if all_ok else PARTIAL, span, recovered, attr_verdicts)


@dataclass
class VerificationReport:
    verdicts: list[ExtractionVerdict]

    @property
    def grounded(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == GROUNDED)

    @property
    def partial(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == PARTIAL)

    @property
    def ungrounded(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == UNGROUNDED)

    @property
    def recovered(self) -> int:
        return sum(1 for v in self.verdicts if v.recovered)

    @property
    def grounding_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return self.grounded / len(self.verdicts)

    def hallucinations(self) -> list[ExtractionVerdict]:
        """What a pipeline should drop or escalate to a human."""
        return [v for v in self.verdicts if v.verdict == UNGROUNDED]


def verify_document(source: str, extractions: list[Extraction]) -> VerificationReport:
    return VerificationReport([verify_extraction(source, e) for e in extractions])
