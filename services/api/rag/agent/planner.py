import json
from typing import Any
from pydantic import BaseModel, ValidationError, Field
from core.config import settings
from rag.llm.openai_compat import chat_completion
from rag.runtime.context import runtime

class Plan(BaseModel):
    intent: str = Field(pattern="^(qa|summarize|compare|quote|troubleshoot)$")
    strategy: str = Field(pattern="^(vector|keyword|hybrid_rrf)$")
    k: int = Field(ge=3, le=50)
    need_rerank: bool
    clarify_if_low_evidence: bool
    response_format: str = Field(pattern="^(normal|bullets|table)$")

def heuristic_plan(user_msg: str, default_k: int) -> Plan:
    q = user_msg.lower()
    intent = "qa"
    if "compare" in q or " vs " in q:
        intent = "compare"
    if "quote" in q or "exact" in q or "where" in q:
        intent = "quote"
    return Plan(
        intent=intent,
        strategy="hybrid_rrf",
        k=default_k,
        need_rerank=False,
        clarify_if_low_evidence=True,
        response_format="normal",
    )

def make_plan(user_msg: str, default_k: int) -> Plan:
    cfg = runtime.provider
    keys = runtime.keys

    base_url = cfg.llm_base_url or settings.default_llm_base_url
    model = cfg.llm_model or settings.default_llm_model
    api_key = keys.llm_api_key

    system = (
        "You are a planner for a RAG system. "
        "Return ONLY valid JSON. No markdown. "
        "Schema: {intent: qa|summarize|compare|quote|troubleshoot, "
        "strategy: vector|keyword|hybrid_rrf, k: int(3..50), need_rerank: bool, "
        "clarify_if_low_evidence: bool, response_format: normal|bullets|table}."
    )
    user = f"User message: {user_msg!r}. Choose the best plan."

    try:
        txt = chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=200,
        )
        data = json.loads(txt)
        return Plan.model_validate(data)
    except Exception:
        return heuristic_plan(user_msg, default_k)
