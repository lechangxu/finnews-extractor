import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from finnews.grounding import (GROUNDED, PARTIAL, UNGROUNDED, Extraction,
                               locate, verify_document, verify_extraction)

SOURCE = "【财联社8月12日电】宁德时代公告称,公司与福特汽车签署战略合作协议,\n将于2027年起交付。"


class TestLocate:
    def test_interval_exact(self):
        text = "宁德时代"
        start = SOURCE.find(text)
        span = locate(SOURCE, text, start, start + len(text))
        assert span is not None and span.method == "interval_exact"
        assert SOURCE[span.start:span.end] == text

    def test_wrong_interval_rejected_then_recovered(self):
        # library claims a wrong interval; verifier must not trust it
        text = "宁德时代"
        span = locate(SOURCE, text, 0, 3)
        assert span is not None and span.method == "exact_search"
        assert SOURCE[span.start:span.end] == text

    def test_null_interval_recovered(self):
        # the real golden-fixture case: library returned char_interval=null
        text = "宁德时代公告称,公司与福特汽车签署战略合作协议"
        span = locate(SOURCE, text, None, None)
        assert span is not None and span.method == "exact_search"
        assert SOURCE[span.start:span.end] == text

    def test_whitespace_wrapped_text(self):
        # extraction omits the newline that exists in the source
        text = "战略合作协议,将于2027年起交付。"
        span = locate(SOURCE, text)
        assert span is not None and span.method == "normalized_search"

    def test_not_found(self):
        assert locate(SOURCE, "比亚迪收购蔚来") is None

    def test_empty_text(self):
        assert locate(SOURCE, "") is None


class TestVerifyExtraction:
    def test_grounded(self):
        ext = Extraction("market_move", "宁德时代", {"code": "宁德时代"})
        v = verify_extraction(SOURCE, ext)
        assert v.verdict == GROUNDED
        assert all(a.grounded for a in v.attributes)

    def test_partial_when_attribute_invented(self):
        ext = Extraction("deal_event", "宁德时代", {"amount": "999亿元"})
        v = verify_extraction(SOURCE, ext)
        assert v.verdict == PARTIAL
        assert not v.attributes[0].grounded

    def test_ungrounded_hallucination(self):
        ext = Extraction("deal_event", "比亚迪收购蔚来", {"company": "比亚迪"})
        v = verify_extraction(SOURCE, ext)
        assert v.verdict == UNGROUNDED
        assert v.span is None

    def test_recovered_flag(self):
        ext = Extraction("deal_event", "宁德时代", {}, char_start=None, char_end=None)
        assert verify_extraction(SOURCE, ext).recovered is True
        ext2 = Extraction("deal_event", "宁德时代", {},
                          char_start=SOURCE.find("宁德时代"),
                          char_end=SOURCE.find("宁德时代") + 4)
        assert verify_extraction(SOURCE, ext2).recovered is False


class TestReport:
    def test_counts_and_hallucination_list(self):
        good = Extraction("market_move", "宁德时代", {})
        bad = Extraction("deal_event", "比亚迪收购蔚来", {})
        report = verify_document(SOURCE, [good, bad])
        assert report.grounded == 1 and report.ungrounded == 1
        assert report.grounding_rate == pytest.approx(0.5)
        assert len(report.hallucinations()) == 1

    def test_empty_report(self):
        report = verify_document(SOURCE, [])
        assert report.grounding_rate == 0.0
