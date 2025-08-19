"""
Long-term memory implementation for persistent knowledge storage using vector search.
"""

import time
import json
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from memory.vector_store import VectorStore


@dataclass
class LongTermMemory:
    """Represents a long-term memory entry."""
    content: str
    memory_type: str  # 'fact', 'preference', 'experience', 'knowledge'
    importance: float  # 0.0 to 1.0
    created_at: float
    last_accessed: float
    access_count: int
    tags: List[str]
    context: Optional[str] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermMemory':
        """Create memory from dictionary."""
        return cls(**data)


class LongTermMemoryManager:
    """
    Manages long-term memory for persistent knowledge storage.
    
    This class uses vector search to store and retrieve memories based on
    semantic similarity, allowing the agent to remember and recall relevant
    information from past conversations.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize long-term memory manager.
        
        Args:
            vector_store: Optional VectorStore instance
        """
        self.vector_store = vector_store or VectorStore()
        self.similarity_threshold = Config.LONG_TERM_MEMORY_THRESHOLD
        self.max_memories = Config.MAX_LONG_TERM_MEMORIES
        
        logger.info("Initialized LongTermMemoryManager")
    
    async def store_memory(self, content: str, memory_type: str = "knowledge",
                          importance: float = 0.5, tags: Optional[List[str]] = None,
                          context: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Store a new long-term memory.
        
        Args:
            content: The memory content
            memory_type: Type of memory ('fact', 'preference', 'experience', 'knowledge')
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags for categorization
            context: Optional context information
            source: Optional source of the memory
            
        Returns:
            The ID of the stored memory
        """
        try:
            # Create memory object
            memory = LongTermMemory(
                content=content,
                memory_type=memory_type,
                importance=max(0.0, min(1.0, importance)),  # Clamp to [0, 1]
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                tags=tags or [],
                context=context,
                source=source
            )
            
            # Prepare metadata
            metadata = memory.to_dict()
            # Remove content from metadata since it's stored separately
            del metadata['content']

            # Convert tags list to string for ChromaDB compatibility
            if 'tags' in metadata and isinstance(metadata['tags'], list):
                metadata['tags'] = ','.join(metadata['tags']) if metadata['tags'] else ''

            # Remove None values and convert to appropriate types for ChromaDB
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)

            metadata = cleaned_metadata
            
            # Store in vector store
            memory_id = await self.vector_store.add_memory(
                text=content,
                metadata=metadata
            )
            
            logger.info(f"Stored long-term memory: {memory_type} - {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")
            raise
    
    async def retrieve_memories(self, query: str, memory_type: Optional[str] = None,
                               max_results: Optional[int] = None,
                               min_importance: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            max_results: Maximum number of results
            min_importance: Minimum importance threshold
            
        Returns:
            List of relevant memory dictionaries
        """
        try:
            max_results = max_results or self.max_memories
            
            # Search for similar memories
            memories = await self.vector_store.search_memories(
                query=query,
                n_results=max_results * 2,  # Get more to allow for filtering
                threshold=self.similarity_threshold
            )

            # If no memories found with threshold, try without threshold
            if not memories:
                memories = await self.vector_store.search_memories(
                    query=query,
                    n_results=max_results,
                    threshold=None  # No threshold - get best matches
                )
            
            # Filter and process results
            filtered_memories = []
            for memory_data in memories:
                metadata = memory_data['metadata']
                
                # Apply filters
                if memory_type and metadata.get('memory_type') != memory_type:
                    continue
                
                if min_importance and metadata.get('importance', 0) < min_importance:
                    continue
                
                # Update access information
                metadata['last_accessed'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                
                # Combine content and metadata
                memory_result = {
                    'id': memory_data['id'],
                    'content': memory_data['text'],
                    'similarity': memory_data['similarity'],
                    **metadata
                }
                
                filtered_memories.append(memory_result)
                
                if len(filtered_memories) >= max_results:
                    break
            
            logger.debug(f"Retrieved {len(filtered_memories)} memories for query: {query[:50]}...")
            return filtered_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def store_conversation_summary(self, summary: str, conversation_context: str,
                                       importance: float = 0.7) -> str:
        """
        Store a conversation summary as a long-term memory.
        
        Args:
            summary: Summary of the conversation
            conversation_context: Context of the conversation
            importance: Importance score
            
        Returns:
            Memory ID
        """
        return await self.store_memory(
            content=summary,
            memory_type="experience",
            importance=importance,
            context=conversation_context,
            source="conversation_summary",
            tags=["conversation", "summary"]
        )
    
    async def store_user_preference(self, preference: str, context: Optional[str] = None) -> str:
        """
        Store a user preference as a long-term memory.
        
        Args:
            preference: The user preference
            context: Optional context
            
        Returns:
            Memory ID
        """
        return await self.store_memory(
            content=preference,
            memory_type="preference",
            importance=0.8,  # Preferences are generally important
            context=context,
            source="user_preference",
            tags=["preference", "user"]
        )
    
    async def store_fact(self, fact: str, context: Optional[str] = None,
                        importance: float = 0.6) -> str:
        """
        Store a factual piece of information.
        
        Args:
            fact: The factual information
            context: Optional context
            importance: Importance score
            
        Returns:
            Memory ID
        """
        return await self.store_memory(
            content=fact,
            memory_type="fact",
            importance=importance,
            context=context,
            source="factual_information",
            tags=["fact", "information"]
        )
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by its ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            Memory dictionary or None
        """
        memory_data = await self.vector_store.get_memory_by_id(memory_id)
        if memory_data:
            # Update access information
            metadata = memory_data['metadata']
            metadata['last_accessed'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            
            return {
                'id': memory_data['id'],
                'content': memory_data['text'],
                **metadata
            }
        return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its ID."""
        return self.vector_store.delete_memory(memory_id)
    
    async def get_memories_by_type(self, memory_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories of a specific type.
        
        Args:
            memory_type: The type of memory to retrieve
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        # Use a broad query to get memories of the specified type
        return await self.retrieve_memories(
            query=memory_type,
            memory_type=memory_type,
            max_results=limit
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about long-term memory."""
        vector_stats = self.vector_store.get_collection_stats()
        return {
            **vector_stats,
            'similarity_threshold': self.similarity_threshold,
            'max_memories_per_query': self.max_memories
        }
    
    def clear_all_memories(self) -> bool:
        """Clear all long-term memories."""
        return self.vector_store.clear_all_memories()
