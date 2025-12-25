from fastapi import APIRouter, HTTPException
from core.models import JobOut
from db.db import get_conn

router = APIRouter()

@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM ingest_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(row)
    finally:
        conn.close()
