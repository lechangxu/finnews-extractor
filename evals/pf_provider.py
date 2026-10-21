"""promptfoo provider over finnews-extractor's grounding verifier.

Two scenarios, selected by the prompt (the yaml test's `scenario` var):

    golden    verify the recorded fixture as-is
    injected  verify it again with one fabricated event appended
              ("比亚迪公告称收购蔚来汽车" — a deal the source never mentions)

Run from the repository root:

    PROMPTFOO_DISABLE_TELEMETRY=1 npx --yes promptfoo@0.124.1 eval \
        -c evals/promptfooconfig.yaml --no-cache --no-write

No network, no API key: the fixture is recorded LangExtract output and the
verifier is pure Python.

`check_results` is the CI gate. `promptfoo eval` exits 100 whenever a case
fails - the same code a crashed worker returns - so green/red is decided by
parsing `results.stats`, and only an exact (2, 0, 0) passes.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "fixtures", "golden_extractions.jsonl")
SCENARIOS = ("golden", "injected")

EXPECTED_SUCCESSES = 2

# The adversarial case: a plausible-looking deal event the source never states.
FAKE_TEXT = "比亚迪公告称收购蔚来汽车"
FAKE_ATTRIBUTES = {
    "company": "比亚迪",
    "counterparty": "蔚来",
    "amount": "500亿元",
    "start": "2026年",
}


def build_payload(scenario, source, extractions, verify_document, extraction_cls) -> dict:
    """Pure assembly: everything finnews-specific is handed in by `call_api`."""
    injected = scenario == "injected"
    extractions = list(extractions)
    if injected:
        extractions = extractions + [
            extraction_cls(
                extraction_class="deal_event",
                extraction_text=FAKE_TEXT,
                attributes=dict(FAKE_ATTRIBUTES),
            )
        ]

    report = verify_document(source, extractions)
    verdicts = report.verdicts
    real = verdicts[:-1] if injected else verdicts

    attribute_pairs = [(a.grounded, a) for v in real for a in v.attributes]
    return {
        "verdicts": [
            {
                "class": v.extraction.extraction_class,
                "verdict": v.verdict,
                "recovered": v.recovered,
            }
            for v in verdicts
        ],
        "grounded": sum(1 for v in real if v.verdict == "grounded"),
        "total_real": len(real),
        "recovered": sum(1 for v in real if v.recovered),
        "attributes_grounded": sum(1 for ok, _ in attribute_pairs if ok),
        "attributes_total": len(attribute_pairs),
        # False unless the injected event really is caught: no credit for silence.
        "injected_flagged": bool(injected and verdicts[-1].verdict == "ungrounded"),
    }


def call_api(prompt, options, context):  # noqa: ARG001 - promptfoo signature
    # Imported here, not at module level, so this module still loads in a
    # minimal environment (the CI gate below runs it without finnews installed).
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from finnews.grounding import Extraction, verify_document
    from finnews.pipeline import load_langextract_jsonl

    scenario = (prompt or "").strip()
    if scenario not in SCENARIOS:
        scenario = (context or {}).get("vars", {}).get("scenario", "golden")

    source, extractions = load_langextract_jsonl(FIXTURE)
    payload = build_payload(scenario, source, extractions, verify_document, Extraction)
    return {
        "output": json.dumps(payload, ensure_ascii=False),
        "metadata": {"scenario": scenario, "fixture": FIXTURE},
    }


def check_results(path: str) -> int:
    """CI gate over a promptfoo `--output` JSON file.

    Prints `successes <n> failures <n> errors <n>` and returns 0 exactly when
    the tuple is (2, 0, 0). Any other outcome is a failure.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        stats = report["results"]["stats"]
        # No int() coercion: 2.9 must not pass as 2, so only an exact match on
        # the reported numbers counts as green.
        successes = stats["successes"]
        failures = stats["failures"]
        errors = stats["errors"]
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and fail
        print(f"could not read promptfoo report {path}: {exc}", file=sys.stderr)
        return 2

    print(f"successes {successes} failures {failures} errors {errors}")
    return 0 if (successes, failures, errors) == (EXPECTED_SUCCESSES, 0, 0) else 1


if __name__ == "__main__":
    raise SystemExit(check_results(sys.argv[1]))
