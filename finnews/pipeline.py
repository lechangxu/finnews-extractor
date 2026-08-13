"""Pipeline glue: load recorded extractions, run live extraction (optional).

Keyless by default — the golden fixture ships in the repo, so the whole
verify/eval/test surface works offline. Live mode (calling an LLM through
LangExtract) is one optional extra and one env var away.
"""

from __future__ import annotations

import json
import os

from .grounding import Extraction, VerificationReport, verify_document
from .schema import PROMPT


def load_langextract_jsonl(path: str) -> tuple[str, list[Extraction]]:
    """Load a LangExtract-annotated JSONL file (one document per line)."""
    with open(path, encoding="utf-8") as f:
        line = f.readline()
    doc = json.loads(line)
    exts = []
    for e in doc["extractions"]:
        ci = e.get("char_interval") or {}
        exts.append(Extraction(
            extraction_class=e["extraction_class"],
            extraction_text=e["extraction_text"],
            attributes=e.get("attributes") or {},
            char_start=ci.get("start_pos"),
            char_end=ci.get("end_pos"),
        ))
    return doc["text"], exts


def verify_jsonl(path: str) -> VerificationReport:
    source, exts = load_langextract_jsonl(path)
    return verify_document(source, exts)


def extract_live(text: str, out_jsonl: str | None = None):
    """Run real extraction via LangExtract + any OpenAI-compatible endpoint.

    Env: FINNEWS_API_KEY (or DEEPSEEK_API_KEY), FINNEWS_BASE_URL,
    FINNEWS_MODEL. Requires the `live` extra: pip install ".[live]".
    """
    try:
        import langextract as lx
        from langextract.providers.openai import OpenAILanguageModel
    except ImportError as e:
        raise RuntimeError(
            'live mode needs the optional extra: pip install ".[live]"') from e

    api_key = os.environ.get("FINNEWS_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("set FINNEWS_API_KEY (or DEEPSEEK_API_KEY) for live mode")

    model = OpenAILanguageModel(
        model_id=os.environ.get("FINNEWS_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.environ.get("FINNEWS_BASE_URL", "https://api.deepseek.com/v1"),
    )
    examples = [
        lx.data.ExampleData(
            text="比亚迪与特斯拉签署电池供应协议, 年供货额约30亿元, 2026年起交付。",
            extractions=[
                lx.data.Extraction(
                    extraction_class="deal_event",
                    extraction_text="比亚迪与特斯拉签署电池供应协议",
                    attributes={"company": "比亚迪", "counterparty": "特斯拉",
                                "amount": "30亿元", "start": "2026年"}),
            ],
        )
    ]
    result = lx.extract(
        text_or_documents=text,
        prompt_description=PROMPT,
        examples=examples,
        model=model,
        fence_output=True,
        use_schema_constraints=False,  # schema constraints are Gemini-only
    )
    if out_jsonl:
        lx.io.save_annotated_documents([result], output_name=out_jsonl)
    return result
