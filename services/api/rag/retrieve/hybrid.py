from typing import Dict, List, Tuple

def rrf_merge(lists: List[List[Tuple[str, float]]], k0: int = 60, top_n: int = 20) -> List[Tuple[str, float]]:
    # lists: each list is ranked high->low by score (already), we use rank only.
    scores: Dict[str, float] = {}
    for lst in lists:
        for rank, (cid, _) in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k0 + rank)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_n]
