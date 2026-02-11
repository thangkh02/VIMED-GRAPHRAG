from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.api.v1.endpoints import search, ingest, graph

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Professional RAG Web Project API"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.app.api.v1.endpoints import search, ingest, graph

app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])

@app.get("/")
def root():
    return {"message": "Welcome to ViMed-GraphRAG API", "docs": "/docs"}
