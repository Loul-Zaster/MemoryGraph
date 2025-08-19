"""
OpenAI embedding utilities for the memory test system.
"""

import asyncio
import sys
import os
from typing import List, Optional
from openai import AsyncOpenAI
from loguru import logger

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class EmbeddingService:
    """Service for generating embeddings using OpenAI's text-embedding-3-small model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding service."""
        self.api_key = api_key or Config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = Config.OPENAI_EMBEDDING_MODEL
        self.dimension = Config.EMBEDDING_DIMENSION
        
        logger.info(f"Initialized EmbeddingService with model: {self.model}")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process in batches to avoid rate limits
            batch_size = 100  # OpenAI's recommended batch size
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, 
                   overlap: Optional[int] = None) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (default from config)
            overlap: Overlap between chunks (default from config)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or Config.CHUNK_SIZE
        overlap = overlap or Config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundaries
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:  # Only if we don't lose too much
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    async def embed_document(self, text: str) -> List[dict]:
        """
        Embed a document by chunking it first.
        
        Args:
            text: Document text to embed
            
        Returns:
            List of dictionaries with 'text' and 'embedding' keys
        """
        chunks = self.chunk_text(text)
        embeddings = await self.embed_texts(chunks)
        
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                'text': chunk,
                'embedding': embedding
            })
        
        logger.info(f"Embedded document into {len(result)} chunks")
        return result


# Utility functions for synchronous usage
def create_embedding_service() -> EmbeddingService:
    """Create an embedding service instance."""
    return EmbeddingService()


async def embed_single_text(text: str) -> List[float]:
    """Utility function to embed a single text."""
    service = create_embedding_service()
    return await service.embed_text(text)
