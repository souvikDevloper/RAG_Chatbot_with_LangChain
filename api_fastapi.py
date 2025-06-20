"""
FastAPI wrapper around the RAG core.
Start with:
    uvicorn api_fastapi:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_core import query, add_document

app = FastAPI(
    title="RAG Chatbot API",
    version="0.1.0",
    description="REST interface for the LangChain-based RAG chatbot",
)

# ---------- request/response models ----------------------------------------
class QuestionIn(BaseModel):
    question: str

class AnswerOut(BaseModel):
    answer: str

class StatusOut(BaseModel):
    status: str
# ---------------------------------------------------------------------------

@app.post("/query", response_model=AnswerOut, summary="Ask the chatbot")
async def ask(question: QuestionIn):
    try:
        answer = query(question.question)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/upload_text",
    response_model=StatusOut,
    summary="Add raw text to the vector store",
)
async def upload_text(text: str):
    try:
        add_document(text)
        return {"status": "document indexed"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/upload_file",
    response_model=StatusOut,
    summary="Add a text file to the vector store",
)
async def upload_file(file: UploadFile = File(...)):
    try:
        content = (await file.read()).decode()
        add_document(content)
        return {"status": f'{file.filename} indexed'}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
