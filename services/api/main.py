from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from db.db import init_db
from routers.collections import router as collections_router
from routers.documents import router as documents_router
from routers.jobs import router as jobs_router
from routers.chat import router as chat_router

app = FastAPI(title="Agentic Hybrid RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok", "data_dir": str(settings.data_dir)}

app.include_router(collections_router, prefix="/collections", tags=["collections"])
app.include_router(documents_router, tags=["documents"])
app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
app.include_router(chat_router, tags=["chat"])
