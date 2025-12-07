from pydantic import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Database settings
    neon_db_url: str = os.getenv("NEON_DB_URL", "")
    
    # Vector database settings
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    # LLM settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    
    # Application settings
    app_name: str = "Book RAG System"
    debug: bool = False
    
    class Config:
        env_file = ".env"

# Create a single instance of settings
settings = Settings()