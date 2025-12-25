from core.config import settings
import chromadb
from chromadb.config import Settings as ChromaSettings

_client = None

def get_client():
    global _client
    if _client is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client

def get_chroma_collection(collection_id: str):
    client = get_client()
    return client.get_or_create_collection(name=f"col_{collection_id}")
