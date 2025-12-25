from typing import List, Tuple
from db.db import get_conn

def keyword_search(collection_id: str, query: str, k: int, document_ids: list[str] | None = None, filename_contains: str | None = None) -> List[Tuple[str, float]]:
    # Returns [(chunk_id, score)] where larger score is better.
    conn = get_conn()
    try:
        q = query.strip()
        if not q:
            return []

        # Basic sanitation for FTS5
        fts_q = " ".join([t for t in q.replace(":", " ").replace("/", " ").split() if t])
        if not fts_q:
            return []

        sql = """
        SELECT f.chunk_id as chunk_id, bm25(chunks_fts) as bm
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH ?
          AND d.collection_id = ?
        """
        params: list = [fts_q, collection_id]

        if document_ids:
            placeholders = ",".join(["?"] * len(document_ids))
            sql += f" AND d.id IN ({placeholders})"
            params.extend(document_ids)

        if filename_contains:
            sql += " AND d.filename LIKE ?"
            params.append(f"%{filename_contains}%")

        sql += " ORDER BY bm LIMIT ?"
        params.append(int(k))

        rows = conn.execute(sql, params).fetchall()
        out: List[Tuple[str, float]] = []
        for r in rows:
            bm = float(r["bm"])
            score = 1.0 / (1.0 + max(0.0, bm))  # invert: smaller bm => higher score
            out.append((r["chunk_id"], score))
        return out
    finally:
        conn.close()
