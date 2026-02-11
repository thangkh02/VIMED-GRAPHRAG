from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter()

@router.post("/")
async def ingest_documents(files: List[UploadFile] = File(...)):
    return {"message": f"Ingested {len(files)} files (Mock)"}
