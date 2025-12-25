from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Chunk:
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    char_start: int
    char_end: int

def chunk_pages(pages: List[tuple[str, Optional[int]]], chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    # pages: list of (text, page_no)
    chunks: List[Chunk] = []
    buf = ""
    buf_pages: List[int] = []
    cursor = 0

    def flush(final: bool = False):
        nonlocal buf, buf_pages, cursor
        if not buf.strip():
            buf = ""
            buf_pages = []
            return
        ps = min(buf_pages) if buf_pages else None
        pe = max(buf_pages) if buf_pages else None
        chunks.append(Chunk(text=buf, page_start=ps, page_end=pe, char_start=cursor, char_end=cursor + len(buf)))
        cursor = cursor + max(0, len(buf) - overlap)
        # keep overlap tail
        if overlap > 0 and len(buf) > overlap and not final:
            buf = buf[-overlap:]
        else:
            buf = ""
        buf_pages = []

    for text, page in pages:
        if not text:
            continue
        # normalize a bit
        t = " ".join(text.split())
        if not t:
            continue
        # append with separator
        if buf:
            buf += "\n"
        start_len = len(buf)
        buf += t
        if page is not None:
            buf_pages.append(page)

        while len(buf) >= chunk_size:
            flush(final=False)

    flush(final=True)
    return chunks
