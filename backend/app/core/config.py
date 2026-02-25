import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ViMed GraphRAG"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    NOTEBOOKS_DIR: str = os.path.join(BASE_DIR, "notebooks")
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = os.path.join(BASE_DIR, "chroma_amg")
    
    # Model Config
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # NLI Verification Config (Self-MedRAG)
    NLI_MODEL_NAME: str = "roberta-large-mnli"
    NLI_SENTENCE_THRESHOLD: float = 0.5
    NLI_PASSAGE_THRESHOLD: float = 0.7
    
    # Groq API Keys (Loaded dynamically)
    GROQ_API_KEYS: List[str] = []

    class Config:
        env_file = ".env"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load GROQ_API_KEY_1 to GROQ_API_KEY_9
        for i in range(1, 10):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key and not key.startswith("gsk_YOUR"):
                self.GROQ_API_KEYS.append(key)

settings = Settings()
