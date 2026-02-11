import time
from typing import List, Optional, Callable, Any, Dict
from langchain_groq import ChatGroq
from backend.app.core.config import settings

class APIKeyManager:
    """Manages rotation of API keys to handle rate limits."""
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_index = 0
        self.failed_keys = set()
    
    def get_current_key(self) -> str:
        if not self.api_keys:
             raise ValueError("No API keys allowed")
        return self.api_keys[self.current_index]
    
    def rotate_key(self) -> bool:
        """Rotates to the next available key. Returns True if successful, False if all keys exhausted."""
        self.failed_keys.add(self.current_index)
        for i in range(len(self.api_keys)):
            next_index = (self.current_index + 1 + i) % len(self.api_keys)
            if next_index not in self.failed_keys:
                self.current_index = next_index
                print(f"Switched to API Key #{next_index + 1}")
                return True
        return False
    
    def reset_failed(self):
        """Resets the failed keys set (e.g., after a waiting period)."""
        self.failed_keys.clear()

class LLMService:
    def __init__(self):
        self.api_manager = APIKeyManager(settings.GROQ_API_KEYS)
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the ChatGroq model with the current key."""
        if not self.api_manager.api_keys:
             print("Warning: No Groq API keys found. LLM disabled.")
             return None
             
        return ChatGroq(
            temperature=0.0,
            model_name=settings.LLM_MODEL,
            api_key=self.api_manager.get_current_key()
        )

    def get_llm(self):
        """Returns the current LLM instance. Re-initializes if needed."""
        if self.llm is None:
             self.llm = self._init_llm()
        return self.llm

    def execute_chain(self, chain_factory: Callable[[Any], Any], input_data: Dict[str, Any], max_retries: int = 3):
        """
        Executes a chain with key rotation logic.
        
        Args:
            chain_factory: A function that takes an LLM instance and returns a Chain/Runnable.
            input_data: Input dictionary for the chain.
            max_retries: Number of retries on RateLimitError.
        """
        for attempt in range(max_retries):
            try:
                # Always get the latest LLM instance
                current_llm = self.get_llm()
                if not current_llm:
                    raise ValueError("LLM not initialized")
                
                # Create the chain using the current LLM
                chain = chain_factory(current_llm)
                return chain.invoke(input_data)
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check for Groq-specific rate limit errors (often 429)
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    print(f"Rate limit hit with key #{self.api_manager.current_index + 1}. Attempting rotation...")
                    
                    if self.api_manager.rotate_key():
                        # Re-init LLM with new key
                        self.llm = self._init_llm()
                        time.sleep(1) # Brief pause
                    else:
                        print("All keys exhausted. Waiting 60s before retry...")
                        time.sleep(60)
                        self.api_manager.reset_failed()
                        self.llm = self._init_llm()
                else:
                    # Not a rate limit error, raise it
                    raise e
                    
        raise Exception("Max retries exceeded for LLM execution")

llm_service = LLMService()
