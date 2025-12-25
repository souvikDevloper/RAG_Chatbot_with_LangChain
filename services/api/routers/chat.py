from fastapi import APIRouter, HTTPException
from core.models import ChatRequest, ChatResponse
from rag.agent.executor import run_agent

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.keys.llm_api_key:
        raise HTTPException(status_code=400, detail="llm_api_key is required (BYOK)")
    if not req.keys.embed_api_key:
        # allow same key if OpenAI-compatible; we'll reuse llm key if embed key missing
        req.keys.embed_api_key = req.keys.llm_api_key

    return run_agent(req)
