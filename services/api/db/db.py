import sqlite3
from pathlib import Path
from core.config import settings

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS collections (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  collection_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  status TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0.0,
  error TEXT,
  provider_json TEXT,   -- MVP: stores provider config for embedding during ingest
  keys_json TEXT,       -- MVP: stores embed key; encrypt in production
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  page_start INTEGER,
  page_end INTEGER,
  section_title TEXT,
  char_start INTEGER,
  char_end INTEGER,
  created_at TEXT NOT NULL
);

-- Keyword search index (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  chunk_id UNINDEXED,
  document_id UNINDEXED,
  filename UNINDEXED,
  tokenize='porter'
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  collection_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    _ensure_parent(settings.sqlite_path)
    conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == col for r in rows)

def init_db() -> None:
    _ensure_parent(settings.sqlite_path)
    conn = get_conn()
    try:
        conn.executescript(SCHEMA_SQL)
        # lightweight migration for older DBs
        if not _col_exists(conn, "ingest_jobs", "provider_json"):
            conn.execute("ALTER TABLE ingest_jobs ADD COLUMN provider_json TEXT")
        if not _col_exists(conn, "ingest_jobs", "keys_json"):
            conn.execute("ALTER TABLE ingest_jobs ADD COLUMN keys_json TEXT")
        conn.commit()
    finally:
        conn.close()
