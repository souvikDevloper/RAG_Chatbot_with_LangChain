from typing import List, Tuple
import requests

def cohere_rerank(api_key: str, query: str, docs: List[Tuple[str, str]], model: str = "rerank-english-v3.0", top_n: int = 8) -> List[Tuple[str, float]]:
    # docs: [(chunk_id, text)]
    url = "https://api.cohere.ai/v1/rerank"
    payload = {
        "model": model,
        "query": query,
        "documents": [{"text": t} for _, t in docs],
        "top_n": min(top_n, len(docs)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    out: List[Tuple[str, float]] = []
    for item in data.get("results", []):
        idx = int(item["index"])
        score = float(item["relevance_score"])
        out.append((docs[idx][0], score))
    return out
