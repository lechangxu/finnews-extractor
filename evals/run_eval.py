#!/usr/bin/env python3
"""Eval harness: grounding verification on the golden fixture + adversarial
hallucination injection. Keyless, deterministic.

    python evals/run_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from finnews.grounding import Extraction, verify_document  # noqa: E402
from finnews.pipeline import load_langextract_jsonl  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "golden_extractions.jsonl")


def main() -> None:
    source, exts = load_langextract_jsonl(FIXTURE)

    # --- adversarial: inject a fabricated event the source never mentions ---
    fake = Extraction(
        extraction_class="deal_event",
        extraction_text="比亚迪公告称收购蔚来汽车",
        attributes={"company": "比亚迪", "counterparty": "蔚来",
                    "amount": "500亿元", "start": "2026年"},
    )
    report = verify_document(source, exts + [fake])

    for v in report.verdicts:
        e = v.extraction
        rec = " recovered" if v.recovered else ""
        print(f"{e.extraction_class:<12} {v.verdict:<10}{rec}  {e.extraction_text[:30]}")

    real = report.verdicts[:-1]
    n_attr = sum(len(v.attributes) for v in real)
    n_attr_ok = sum(sum(1 for a in v.attributes if a.grounded) for v in real)
    fake_flagged = report.verdicts[-1].verdict == "ungrounded"

    print()
    print(f"real extractions grounded : {report.grounded}/{len(real)}")
    print(f"library-alignment failures recovered : "
          f"{sum(1 for v in real if v.recovered)}")
    print(f"attribute values grounded : {n_attr_ok}/{n_attr}")
    print(f"injected hallucination flagged : {'yes' if fake_flagged else 'NO — BUG'}")

    ok = (report.grounded == len(real) and n_attr_ok == n_attr and fake_flagged)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
