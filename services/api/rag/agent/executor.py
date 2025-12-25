from typing import List, Tuple

from core.config import settings
from core.models import ChatRequest, ChatResponse, Citation, ChatTrace, ChatTraceStep
from db.db import get_conn

from rag.runtime.context import set_runtime
from rag.agent.planner import make_plan
from rag.agent.router import route_intent
from rag.agent.citations import enforce_citations, extract_citations

from rag.llm.openai_compat import chat_completion
from rag.llm.embeddings import embed_texts

from rag.retrieve.keyword import keyword_search
from rag.retrieve.vector import vector_search
from rag.retrieve.hybrid import rrf_merge
from rag.retrieve.rerank import cohere_rerank

def _fetch_chunks(chunk_ids: List[str]) -> List[dict]:
    if not chunk_ids:
        return []
    conn = get_conn()
    try:
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = conn.execute(
            f"""
            SELECT c.id as chunk_id, c.text, c.page_start, c.page_end,
                   d.id as document_id, d.filename
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        by_id = {r["chunk_id"]: dict(r) for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]
    finally:
        conn.close()

def run_agent(req: ChatRequest) -> ChatResponse:
    set_runtime(req.provider, req.keys)

    trace = ChatTrace(intent="qa", strategy="hybrid_rrf", steps=[])

    k = req.options.k or settings.default_top_k
    plan = make_plan(req.message, default_k=k)
    intent = route_intent(plan.intent)
    trace.intent = intent
    trace.strategy = plan.strategy
    trace.steps.append(ChatTraceStep(tool="plan", meta=plan.model_dump()))

    q_emb = None
    if plan.strategy in {"vector", "hybrid_rrf"}:
        q_emb = embed_texts([req.message])[0]

    kw_list: List[Tuple[str, float]] = []
    vec_list: List[Tuple[str, float]] = []

    if plan.strategy in {"keyword", "hybrid_rrf"}:
        kw_list = keyword_search(
            collection_id=req.collection_id,
            query=req.message,
            k=plan.k,
            document_ids=req.filters.document_ids or None,
            filename_contains=req.filters.filename_contains,
        )
        kw_list = sorted(kw_list, key=lambda x: x[1], reverse=True)
        trace.steps.append(ChatTraceStep(tool="keyword_search", meta={"k": plan.k, "hits": len(kw_list)}))

    if plan.strategy in {"vector", "hybrid_rrf"} and q_emb is not None:
        vec_list = vector_search(
            collection_id=req.collection_id,
            query_emb=q_emb,
            k=plan.k,
            document_ids=req.filters.document_ids or None,
            filename_contains=req.filters.filename_contains,
        )
        vec_list = sorted(vec_list, key=lambda x: x[1], reverse=True)
        trace.steps.append(ChatTraceStep(tool="vector_search", meta={"k": plan.k, "hits": len(vec_list)}))

    if plan.strategy == "keyword":
        merged = kw_list[: max(8, plan.k)]
    elif plan.strategy == "vector":
        merged = vec_list[: max(8, plan.k)]
    else:
        merged = rrf_merge([kw_list, vec_list], k0=60, top_n=max(10, plan.k))
        trace.steps.append(ChatTraceStep(tool="rrf_merge", meta={"top_n": max(10, plan.k)}))

    score_map = {cid: float(s) for cid, s in merged}

    chunk_ids = [cid for cid, _ in merged]
    chunks = _fetch_chunks(chunk_ids)

    if not chunks:
        ans = "I couldn't find relevant information in the uploaded documents. Which document or topic should I focus on?"
        return ChatResponse(answer=ans, citations=[], agent_trace=trace)

    # Optional rerank (Cohere)
    if req.options.use_rerank and req.keys.rerank_api_key and req.provider.rerank_provider == "cohere":
        docs = [(c["chunk_id"], c["text"]) for c in chunks]
        model = req.provider.rerank_model or "rerank-english-v3.0"
        reranked = cohere_rerank(
            api_key=req.keys.rerank_api_key,
            query=req.message,
            docs=docs,
            model=model,
            top_n=min(8, len(docs)),
        )
        rid_order = [cid for cid, _ in reranked]
        chunks = _fetch_chunks(rid_order)
        trace.steps.append(ChatTraceStep(tool="rerank", meta={"provider": "cohere", "model": model, "top_n": len(chunks)}))

    # Build context
    context_blocks = []
    initial_citations: List[Citation] = []
    for i, c in enumerate(chunks[: min(len(chunks), 12)]):
        context_blocks.append(
            f"""[CHUNK {i+1}]
chunk_id={c['chunk_id']}
filename={c['filename']}
pages={c['page_start']}-{c['page_end']}
text={c['text']}
"""
        )
        initial_citations.append(
            Citation(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                filename=c["filename"],
                page_start=c["page_start"],
                page_end=c["page_end"],
                score=score_map.get(c["chunk_id"], 0.0),
            )
        )

    sys = (
        "You are a helpful assistant answering strictly from the provided context. "
        "If the answer is not in the context, say: 'Not found in documents.' "
        "Cite sources by adding tags like [[cid:<chunk_id>]] next to the sentence they support. "
        "Do not invent citations."
    )

    if intent == "summarize":
        user_prefix = "Summarize the relevant parts of the documents for the user's request."
    elif intent == "compare":
        user_prefix = "Compare the relevant parts and present differences clearly. Use a small table if helpful."
    elif intent == "quote":
        user_prefix = "Find and quote the exact relevant text. Keep quotes short and cite the chunk."
    else:
        user_prefix = "Answer the user's question."

    user = user_prefix + "\n\nUSER QUESTION:\n" + req.message + "\n\nCONTEXT:\n" + "\n".join(context_blocks)

    base_url = req.provider.llm_base_url or settings.default_llm_base_url
    model = req.provider.llm_model or settings.default_llm_model

    ans = chat_completion(
        base_url=base_url,
        api_key=req.keys.llm_api_key,
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=900,
    )
    trace.steps.append(ChatTraceStep(tool="answer", meta={"model": model}))

    top_cid = chunks[0]["chunk_id"] if chunks else None
    valid_cids = {c["chunk_id"] for c in chunks}
    ans2 = enforce_citations(ans, top_cid, valid_cids)

    cited_ids = set(extract_citations(ans2))
    citations_out = [c for c in initial_citations if c.chunk_id in cited_ids]
    if not citations_out:
        citations_out = initial_citations[:3]

    return ChatResponse(answer=ans2, citations=citations_out, agent_trace=trace)
