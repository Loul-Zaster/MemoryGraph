"""
Configuration settings for the Multi-Agent Memory Test System.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the memory test system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o-mini"  # Main chat model
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # Recommended embedding model
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "memory_collection"
    
    # Memory Configuration
    SHORT_TERM_MEMORY_SIZE: int = 10  # Number of recent messages to keep
    LONG_TERM_MEMORY_THRESHOLD: float = 0.2  # Similarity threshold for retrieval
    MAX_LONG_TERM_MEMORIES: int = 5  # Max memories to retrieve
    
    # Agent Configuration
    AGENT_NAME: str = "MemoryAgent"
    AGENT_DESCRIPTION: str = "An AI agent with both short-term and long-term memory capabilities"
    
    # Embedding Configuration
    EMBEDDING_DIMENSION: int = 1536  # Dimension for text-embedding-3-small
    CHUNK_SIZE: int = 1000  # Text chunk size for embeddings
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in a .env file or environment variable.")
            return False
        return True
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Get OpenAI configuration dictionary."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "embedding_model": cls.OPENAI_EMBEDDING_MODEL
        }
    
    @classmethod
    def get_chroma_config(cls) -> dict:
        """Get ChromaDB configuration dictionary."""
        return {
            "persist_directory": cls.CHROMA_DB_PATH,
            "collection_name": cls.CHROMA_COLLECTION_NAME
        }
    
    @classmethod
    def get_memory_config(cls) -> dict:
        """Get memory configuration dictionary."""
        return {
            "short_term_size": cls.SHORT_TERM_MEMORY_SIZE,
            "long_term_threshold": cls.LONG_TERM_MEMORY_THRESHOLD,
            "max_long_term_memories": cls.MAX_LONG_TERM_MEMORIES
        }
