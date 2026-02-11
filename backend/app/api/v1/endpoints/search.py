from fastapi import APIRouter, HTTPException
from backend.app.models.schemas import SearchRequest, SearchResponse
from backend.app.services.rag_service import rag_service

router = APIRouter()

@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest):
    try:
        # For now, simplistic mapping
        result = await rag_service.process_question(request.query)
        return SearchResponse(results=[str(result)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
