from typing import List, Optional
from pydantic import BaseModel

class EntityBase(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    relevance_score: Optional[int] = None

class RelationBase(BaseModel):
    source: str
    target: str
    type: str
    confidence_score: Optional[int] = None
    evidence: Optional[str] = None

class ExtractionRequest(BaseModel):
    text: str

class ExtractionResponse(BaseModel):
    entities: List[EntityBase]
    relations: List[RelationBase]
    
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[str] # Simplified for now
    
class GraphResponse(BaseModel):
    html_content: str
