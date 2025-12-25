from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)

class CollectionOut(BaseModel):
    id: str
    name: str
    created_at: str

class DocumentOut(BaseModel):
    id: str
    collection_id: str
    filename: str
    status: str
    created_at: str

class JobOut(BaseModel):
    id: str
    document_id: str
    status: str
    progress: float
    error: Optional[str]
    created_at: str
    updated_at: str

class ProviderConfig(BaseModel):
    # OpenAI-compatible base URL. Works for OpenAI, Groq, OpenRouter, etc.
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    embed_base_url: Optional[str] = None
    embed_model: Optional[str] = None

    rerank_provider: Optional[Literal["cohere"]] = None
    rerank_model: Optional[str] = None

class KeysConfig(BaseModel):
    llm_api_key: Optional[str] = None
    embed_api_key: Optional[str] = None
    rerank_api_key: Optional[str] = None

class ChatFilters(BaseModel):
    document_ids: List[str] = []
    filename_contains: Optional[str] = None

class ChatOptions(BaseModel):
    k: Optional[int] = None
    use_rerank: bool = False

class ChatRequest(BaseModel):
    collection_id: str
    session_id: str
    message: str
    filters: ChatFilters = ChatFilters()
    provider: ProviderConfig = ProviderConfig()
    keys: KeysConfig = KeysConfig()
    options: ChatOptions = ChatOptions()

class Citation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: float

class ChatTraceStep(BaseModel):
    tool: str
    meta: Dict[str, Any] = {}

class ChatTrace(BaseModel):
    intent: str
    strategy: str
    steps: List[ChatTraceStep] = []

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    agent_trace: ChatTrace
