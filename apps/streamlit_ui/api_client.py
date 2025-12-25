import os
import requests
from typing import Any, Dict

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def _url(path: str) -> str:
    return API_BASE_URL.rstrip("/") + path

def create_collection(name: str) -> Dict[str, Any]:
    r = requests.post(_url("/collections"), json={"name": name}, timeout=30)
    r.raise_for_status()
    return r.json()

def list_collections():
    r = requests.get(_url("/collections"), timeout=30)
    r.raise_for_status()
    return r.json()

def upload_document(collection_id: str, file_bytes: bytes, filename: str, embed_api_key: str, embed_base_url: str, embed_model: str):
    files = {"file": (filename, file_bytes)}
    data = {
        "embed_api_key": embed_api_key,
        "embed_base_url": embed_base_url,
        "embed_model": embed_model,
    }
    r = requests.post(_url(f"/collections/{collection_id}/documents"), files=files, data=data, timeout=300)
    r.raise_for_status()
    return r.json()

def list_documents(collection_id: str):
    r = requests.get(_url(f"/collections/{collection_id}/documents"), timeout=30)
    r.raise_for_status()
    return r.json()

def get_job(job_id: str):
    r = requests.get(_url(f"/jobs/{job_id}"), timeout=30)
    r.raise_for_status()
    return r.json()

def chat(payload: Dict[str, Any]):
    r = requests.post(_url("/chat"), json=payload, timeout=180)
    r.raise_for_status()
    return r.json()
