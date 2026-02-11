from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from backend.app.services.graph_service import graph_service
from backend.app.core.config import settings
import os

router = APIRouter()

@router.get("/visualize", response_class=HTMLResponse)
async def get_graph_visualization():
    # Helper to retrieve the latest graph
    # In a real app, this might accept a query ID
    file_path = os.path.join(settings.DATA_DIR, "current_graph.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return "<h1>No graph available</h1>"
