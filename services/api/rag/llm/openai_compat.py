from typing import List, Dict, Any, Optional
import requests

def chat_completion(base_url: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=90)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def embeddings(base_url: str, api_key: str, model: str, texts: List[str]) -> List[List[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    j = r.json()
    return [d["embedding"] for d in j["data"]]
