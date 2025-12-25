from typing import List
from core.config import settings
from rag.llm.openai_compat import embeddings as _emb
from rag.runtime.context import runtime

def embed_texts(texts: List[str]) -> List[List[float]]:
    cfg = runtime.provider
    keys = runtime.keys
    base_url = cfg.embed_base_url or settings.default_embed_base_url
    model = cfg.embed_model or settings.default_embed_model
    api_key = keys.embed_api_key
    if not api_key:
        raise ValueError("embed_api_key missing")
    return _emb(base_url=base_url, api_key=api_key, model=model, texts=texts)
