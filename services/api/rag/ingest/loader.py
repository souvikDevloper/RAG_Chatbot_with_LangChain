from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import csv

from pypdf import PdfReader
from docx import Document as DocxDocument

@dataclass
class PageText:
    text: str
    page: Optional[int] = None

def load_text(path: str) -> List[PageText]:
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".pdf":
        reader = PdfReader(str(p))
        out: List[PageText] = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            out.append(PageText(text=t, page=i + 1))
        return out

    if suf in {".txt", ".md"}:
        return [PageText(text=p.read_text(encoding="utf-8", errors="ignore"), page=None)]

    if suf == ".csv":
        rows = []
        with open(p, "r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.reader(f)
            for row in r:
                rows.append(", ".join(row))
        return [PageText(text="\n".join(rows), page=None)]

    if suf == ".docx":
        doc = DocxDocument(str(p))
        paras = [para.text for para in doc.paragraphs]
        return [PageText(text="\n".join(paras), page=None)]

    # fallback: best effort text
    return [PageText(text=p.read_text(encoding="utf-8", errors="ignore"), page=None)]
