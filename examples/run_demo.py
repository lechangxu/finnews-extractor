#!/usr/bin/env python3
"""Keyless demo: verify the recorded golden extractions against the source.

    python examples/run_demo.py          # offline, uses the golden fixture
    python examples/run_demo.py --live   # calls the LLM (needs FINNEWS_API_KEY)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from finnews.grounding import GROUNDED, PARTIAL  # noqa: E402
from finnews.pipeline import verify_jsonl  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "golden_extractions.jsonl")

_BADGE = {GROUNDED: "✓ grounded", PARTIAL: "~ partial", "ungrounded": "✗ UNGROUNDED"}


def main() -> None:
    if "--live" in sys.argv:
        from finnews.pipeline import extract_live
        text = open(os.path.join(os.path.dirname(__file__), "..",
                                 "corpus", "sample_news.txt"), encoding="utf-8").read()
        result = extract_live(text, out_jsonl="live_output.jsonl")
        print(f"live extraction: {len(result.extractions)} events "
              f"→ saved to live_output.jsonl, now verify it:")
        report = verify_jsonl("live_output.jsonl")
    else:
        report = verify_jsonl(FIXTURE)

    for v in report.verdicts:
        e = v.extraction
        span = f"chars {v.span.start}–{v.span.end} ({v.span.method})" if v.span else "—"
        tag = " [recovered: library alignment failed]" if v.recovered else ""
        print(f"{_BADGE[v.verdict]:<13} {e.extraction_class}{tag}")
        print(f"              text: {e.extraction_text}")
        print(f"              span: {span}")
        for a in v.attributes:
            mark = "✓" if a.grounded else "✗ not in source"
            print(f"              {a.name}: {a.value}  {mark}")
    print()
    print(f"grounded {report.grounded}/{len(report.verdicts)} "
          f"(rate {report.grounding_rate:.0%}), "
          f"recovered {report.recovered}, hallucinations {report.ungrounded}")


if __name__ == "__main__":
    main()
