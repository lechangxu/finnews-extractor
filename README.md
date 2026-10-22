# finnews-extractor

[![tests](https://github.com/lechangxu/finnews-extractor/actions/workflows/tests.yml/badge.svg)](https://github.com/lechangxu/finnews-extractor/actions/workflows/tests.yml)

**Chinese financial news → structured events, with independently verified source grounding.** Every extracted event must point back to where it lives in the original text — and every attribute value must exist in the source — or it gets flagged as a hallucination.

## Why this exists

LLM extraction pipelines produce structured data you can't trust by default. Tools like [LangExtract](https://github.com/google/langextract) *claim* a source span for each extraction — but the claim itself is model output, and it fails silently. Real example from this repo's golden fixture (a recorded DeepSeek run): **all three events came back with failed or fuzzy alignment — one with `char_interval: null` — while every extraction was in fact verbatim in the source.** This library re-verifies independently:

| | extraction with claimed span | span independently re-verified | attribute values checked against source | runs keyless, CI-friendly |
|---|---|---|---|---|
| LangExtract alone | ✅ | — (trusts its own alignment) | — | — (needs an LLM key) |
| **finnews-extractor** | ✅ (via LangExtract, optional) | **✅ — recovers spans the library failed to align** | **✅ — per attribute** | **✅** |

Same philosophy as my [rag-with-receipts](https://github.com/lechangxu/rag-with-receipts) (verify every *answer* sentence against its citation), applied one pipeline stage earlier: verify every *extraction* against the source text. Neither trusts the model's word.

## Quickstart (keyless)

```bash
pip install -e ".[dev]"
pytest                      # 16 tests
python examples/run_demo.py # verify the recorded golden fixture
python evals/run_eval.py    # grounding eval + adversarial hallucination injection
```

No API key needed: the repo ships a golden fixture — real recorded LLM output on a synthetic A-share news wire — and the entire verification layer is deterministic and offline.

Live extraction (optional): `pip install ".[live]"`, set `FINNEWS_API_KEY` (any OpenAI-compatible endpoint; defaults to DeepSeek), then `python examples/run_demo.py --live`.

## Demo output (real, this commit)

```
✓ grounded    deal_event [recovered: library alignment failed]
              text: 宁德时代公告称,公司与福特汽车签署战略合作协议
              span: chars 11–34 (exact_search)
              company: 宁德时代  ✓
              counterparty: 福特汽车  ✓
              amount: 45亿元人民币  ✓
              start: 2027年  ✓
...
grounded 3/3 (rate 100%), recovered 3, hallucinations 0
```

Eval harness (golden fixture + one injected fabrication `比亚迪公告称收购蔚来汽车`):

```
real extractions grounded : 3/3
library-alignment failures recovered : 3
attribute values grounded : 11/11
injected hallucination flagged : yes
PASS
```

Honest framing: this is a smoke fixture (one document, three events) proving the machinery — span recovery, attribute grounding, hallucination flagging — works end to end. The harness is the product: point it at your own recorded extractions and you have a regression suite for your extraction pipeline.

## How to measure the hallucination rate

The verifier already records which extractions it could not ground, so the headline number is free:

```
hallucination rate = ungrounded extractions / all extractions
```

Two numbers decide whether that rate means anything:

- **Detection rate** — the share of *injected* fake events the verifier catches. A verifier that never flags anything would score 0 here.
- **False-positive rate** — the share of *real* events it wrongly flags. A verifier that flags everything would have perfect detection and a false-positive rate of 100% — useless all the same, which is why the two numbers only mean something together.

On this fixture both are perfect: detection is 1/1 (100%) — the one injected fake is flagged — and the false-positive rate is 0/3 (0%) — all three real events pass. Those two hypotheticals above are the failure modes the suite is built to catch, not measurements of this verifier.

`evals/promptfooconfig.yaml` turns both into a declarative [promptfoo](https://www.promptfoo.dev/) suite over the recorded fixture: the golden run, plus one adversarial run with a fabricated deal (`比亚迪公告称收购蔚来汽车`) appended to the real extractions.

```
PROMPTFOO_DISABLE_TELEMETRY=1 npx --yes promptfoo@0.124.1 eval -c evals/promptfooconfig.yaml --no-cache --no-write
```

Real output on this commit:

```
✓ 2 passed (100%)
0 failed (0%)
0 errors (0%)
```

Per scenario, read out of the provider's JSON:

```
real extractions grounded : 3/3
attribute values grounded : 11/11
injected hallucination flagged : yes
```

`promptfoo eval` exits 100 whenever a case fails — the same code a crashed worker returns — so this repo does not gate on the exit code. `python3 evals/pf_provider.py <results.json>` parses `results.stats` and passes only on exactly 2 successes, 0 failures, 0 errors; CI runs both steps.

This is a smoke fixture (one document), not a benchmark.

## How verification works

1. **Trust-but-verify the claimed interval** — if the library's `char_interval` slices exactly to the extraction text, done (`interval_exact`).
2. **Recover** — otherwise search the source: exact substring, then whitespace-normalized substring mapped back to original offsets (`exact_search` / `normalized_search`). A recovered span means the model was right and the aligner was wrong.
3. **Ground every attribute** — each attribute value must appear in the source (exact or whitespace-normalized). Invented amounts, dates, tickers → `partial`.
4. **Flag the rest** — extraction text not found anywhere → `ungrounded`. That's your hallucination list: drop it or escalate it.

Verdicts are data, not policy: your pipeline decides whether `partial` means drop, fix, or human-review.

## Stack

| Layer | What it is |
|---|---|
| Language | Python 3.10+ (`requires-python` in `pyproject.toml`) |
| Install | `pip install -e ".[dev]"` |
| Runtime dependencies | none — standard library only |
| Optional extra | `langextract >= 0.4` for live extraction (`pip install -e ".[live]"`) |
| Tests | pytest — 16 tests in `tests/` |
| CI | GitHub Actions, Python 3.11 + 3.12, plus a promptfoo eval gate |

## Layout

| file | role |
|---|---|
| `finnews/grounding.py` | **the verifier** — span recovery, attribute grounding, verdicts |
| `finnews/schema.py` | 3 event classes for Chinese financial wires (deal / market move / macro action) |
| `finnews/pipeline.py` | load recorded LangExtract output; optional live extraction |
| `fixtures/golden_extractions.jsonl` | real recorded LLM output — the offline test bed |
| `corpus/sample_news.txt` | synthetic A-share style news wire |
| `evals/run_eval.py` | runnable eval incl. adversarial hallucination injection |
| `docs/demo_visualization.html` | LangExtract's own visualization of the same fixture |

## License

MIT. Use it in client work; attribution appreciated.
