from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import shutil
import os
from backend.app.core.config import settings
from backend.app.services.rag_service import rag_service

router = APIRouter()

@router.post("/")
async def ingest_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload PDFs and start ingestion process in background"""
    saved_files = []
    
    # Ensure data dir exists
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
            
        file_path = os.path.join(settings.DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
        
        # Add to background task
        background_tasks.add_task(rag_service.ingest_document, file_path)
    
    return {
        "message": f"Received {len(saved_files)} PDF files. Processing started in background.",
        "files": [os.path.basename(f) for f in saved_files]
    }
