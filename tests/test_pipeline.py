import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from finnews.pipeline import (extract_live, load_langextract_jsonl,  # noqa: E402
                              verify_jsonl)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "golden_extractions.jsonl")


class TestFixture:
    def test_loads_real_recorded_output(self):
        source, exts = load_langextract_jsonl(FIXTURE)
        assert "宁德时代" in source
        assert len(exts) == 3
        classes = [e.extraction_class for e in exts]
        assert classes == ["deal_event", "market_move", "macro_action"]

    def test_golden_all_grounded(self):
        report = verify_jsonl(FIXTURE)
        assert report.grounded == 3
        assert report.ungrounded == 0
        # the deal_event's null interval was recovered by exact search
        assert report.recovered >= 1

    def test_golden_attributes_all_grounded(self):
        report = verify_jsonl(FIXTURE)
        for v in report.verdicts:
            for a in v.attributes:
                assert a.grounded, f"{a.name}={a.value} not in source"


class TestLiveModeGuards:
    def test_live_without_key_raises_cleanly(self, monkeypatch):
        monkeypatch.delenv("FINNEWS_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        try:
            import langextract  # noqa: F401
            with pytest.raises(RuntimeError, match="FINNEWS_API_KEY"):
                extract_live("test")
        except ImportError:
            with pytest.raises(RuntimeError, match="pip install"):
                extract_live("test")
