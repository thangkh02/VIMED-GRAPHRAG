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


# ---------------------------------------------------------------------------
# NLI Verification Schemas (Self-MedRAG)
# ---------------------------------------------------------------------------

class StatementVerification(BaseModel):
    """Verification result for a single rationale statement."""
    statement: str
    label: str  # "Supported" or "Unsupported"
    confidence_score: float
    best_passage: Optional[str] = None


class VerificationRequest(BaseModel):
    """Input for the SelfReflectiveCritic verification pipeline."""
    statements: List[str]
    passages: List[str]


class VerificationResponse(BaseModel):
    """Output of the SelfReflectiveCritic verification pipeline."""
    is_passed: bool
    support_score: float
    supported_statements: List[StatementVerification]
    unsupported_statements: List[StatementVerification]
