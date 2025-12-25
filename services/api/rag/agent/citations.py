import re
from typing import List, Set

CITE_RE = re.compile(r"\[\[cid:([0-9a-fA-F\-]{36})\]\]")

def extract_citations(text: str) -> List[str]:
    return CITE_RE.findall(text)

def enforce_citations(answer: str, top_chunk_id: str | None, valid_chunk_ids: Set[str] | None = None) -> str:
    # Each paragraph must include at least one [[cid:...]] tag.
    valid = valid_chunk_ids or set()
    paras = [p for p in answer.split("\n\n") if p.strip()]
    fixed = []
    for p in paras:
        def _replace(m: re.Match) -> str:
            cid = m.group(1)
            if not valid or cid in valid:
                return m.group(0)
            return f"[[cid:{top_chunk_id}]]" if top_chunk_id else ""

        p2 = CITE_RE.sub(_replace, p)
        if CITE_RE.search(p2):
            fixed.append(p2)
        else:
            if top_chunk_id:
                fixed.append(p2.rstrip() + f" [[cid:{top_chunk_id}]]")
            else:
                fixed.append("Not found in documents.")
    return "\n\n".join(fixed)
