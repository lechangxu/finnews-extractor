"""Extraction schema for Chinese financial news: what events to pull out.

Three event classes cover the bulk of A-share style news wires. Each class
declares the attributes a downstream system can rely on — and every
attribute value is independently checked against the source by
`finnews.grounding`.
"""

SCHEMA = {
    "deal_event": {
        "description": "公司间合作/订单/协议事件 (inter-company deal or order)",
        "attributes": ["company", "counterparty", "amount", "start"],
    },
    "market_move": {
        "description": "个股异动 (single-stock price move)",
        "attributes": ["code", "change_percent", "current_price"],
    },
    "macro_action": {
        "description": "央行/政策操作 (central-bank or policy operation)",
        "attributes": ["institution", "tool", "scale", "rate"],
    },
}

PROMPT = """从财经新闻中抽取结构化信息:
1. deal_event: 公司间合作/订单事件 (提取 公司、对手方、金额、时间)
2. market_move: 个股异动 (代码、涨跌幅、现价)
3. macro_action: 央行/政策操作 (机构、工具、规模、利率)
按出现顺序抽取, 不要改写原文措辞。"""
