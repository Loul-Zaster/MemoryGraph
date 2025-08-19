"""
ChromaDB vector store implementation for memory management.
"""

import os
import uuid
import sys
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from loguru import logger

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.embeddings import EmbeddingService


class VectorStore:
    """ChromaDB-based vector store for semantic search and memory storage."""

    def __init__(self, persist_directory: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        """Initialize the vector store."""
        self.persist_directory = persist_directory or Config.CHROMA_DB_PATH
        self.user_id = user_id
        self.session_id = session_id

        # Generate collection name based on user/session if provided
        if user_id and session_id:
            self.collection_name = f"user_{user_id}_session_{session_id}"
        elif user_id:
            self.collection_name = f"user_{user_id}_global"
        else:
            self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Memory storage for AI agent"}
        )
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        logger.info(f"Initialized VectorStore with collection: {self.collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
    
    async def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                        memory_id: Optional[str] = None) -> str:
        """
        Add a memory to the vector store.
        
        Args:
            text: The memory text to store
            metadata: Optional metadata associated with the memory
            memory_id: Optional custom ID for the memory
            
        Returns:
            The ID of the stored memory
        """
        try:
            # Generate embedding for the text
            embedding = await self.embedding_service.embed_text(text)
            
            # Generate ID if not provided
            if memory_id is None:
                memory_id = str(uuid.uuid4())
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add timestamp if not present
            if 'timestamp' not in metadata:
                import time
                metadata['timestamp'] = time.time()
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            logger.info(f"Added memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    async def search_memories(self, query: str, n_results: int = 5,
                             threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar memories using semantic search.
        
        Args:
            query: The search query
            n_results: Number of results to return
            threshold: Similarity threshold (optional)
            
        Returns:
            List of memory dictionaries with text, metadata, and similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search in the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity = 1 - distance
                    
                    # Apply threshold if specified
                    if threshold is not None and similarity < threshold:
                        continue
                    
                    memory = {
                        'text': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    memories.append(memory)
            
            logger.debug(f"Found {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by its ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            Memory dictionary or None if not found
        """
        try:
            results = self.collection.get(
                ids=[memory_id],
                include=['documents', 'metadatas']
            )
            
            if results['documents'] and results['documents'][0]:
                return {
                    'id': memory_id,
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory with ID: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_memories': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def clear_all_memories(self) -> bool:
        """
        Clear all memories from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Memory storage for AI agent"}
            )
            
            logger.warning("Cleared all memories from the collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False
