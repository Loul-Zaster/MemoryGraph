"""
Memory-enabled AI agent with both short-term and long-term memory capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from loguru import logger
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemoryManager
from memory.vector_store import VectorStore
from utils.session_manager import SessionManager


class MemoryAgent:
    """
    AI agent with integrated memory capabilities.
    
    This agent combines short-term conversation context with long-term
    semantic memory to provide contextually aware responses.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        """
        Initialize the memory agent.

        Args:
            api_key: OpenAI API key (optional, will use config if not provided)
            user_id: User ID for memory isolation
            session_id: Session ID for memory isolation
        """
        # Initialize OpenAI client
        self.api_key = api_key or Config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=Config.OPENAI_MODEL,
            temperature=0.7
        )

        # Store user and session info
        self.user_id = user_id
        self.session_id = session_id

        # Initialize memory systems with user/session isolation
        self.short_term_memory = ShortTermMemory()
        self.vector_store = VectorStore(user_id=user_id, session_id=session_id)
        self.long_term_memory = LongTermMemoryManager(self.vector_store)
        
        # Agent configuration
        self.agent_name = Config.AGENT_NAME
        self.agent_description = Config.AGENT_DESCRIPTION
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"Initialized {self.agent_name}")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return f"""You are {self.agent_name}, {self.agent_description}.

You have access to both short-term and long-term memory:

SHORT-TERM MEMORY: Contains the recent conversation context. Use this to maintain conversation flow and refer to recent topics.

LONG-TERM MEMORY: Contains persistent knowledge, facts, preferences, and experiences from past conversations. Use this to:
- Remember user preferences and personal information
- Recall relevant facts and knowledge from previous interactions
- Provide continuity across conversation sessions
- Learn from past experiences

MEMORY GUIDELINES:
1. Always consider both short-term context and relevant long-term memories when responding
2. When you learn something important about the user, suggest storing it in long-term memory
3. If the user mentions something that contradicts stored memories, ask for clarification
4. Use memories to personalize responses and show continuity
5. Be transparent about what you remember and what you don't

RESPONSE STYLE:
- Be helpful, conversational, and engaging
- Show that you remember previous interactions when relevant
- Ask clarifying questions when needed
- Suggest storing important information for future reference

Remember: Your goal is to provide a seamless, personalized experience by effectively using both types of memory."""
    
    async def process_message(self, user_message: str, 
                             store_in_memory: bool = True) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: The user's message
            store_in_memory: Whether to store the interaction in memory
            
        Returns:
            Dictionary containing response and memory information
        """
        try:
            # Add user message to short-term memory
            if store_in_memory:
                self.short_term_memory.add_user_message(user_message)
            
            # Retrieve relevant long-term memories
            relevant_memories = await self.long_term_memory.retrieve_memories(
                query=user_message,
                max_results=Config.MAX_LONG_TERM_MEMORIES
            )
            
            # Prepare context for the LLM
            context = self._prepare_context(user_message, relevant_memories)
            
            # Generate response
            response = await self._generate_response(context)
            
            # Add assistant response to short-term memory
            if store_in_memory:
                self.short_term_memory.add_assistant_message(response)
            
            # Analyze if we should store anything in long-term memory
            memory_suggestions = await self._analyze_for_memory_storage(
                user_message, response, relevant_memories
            )
            
            return {
                'response': response,
                'relevant_memories': relevant_memories,
                'memory_suggestions': memory_suggestions,
                'short_term_context': self.short_term_memory.get_conversation_context()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'response': "I apologize, but I encountered an error processing your message. Please try again.",
                'error': str(e)
            }
    
    def _prepare_context(self, user_message: str, 
                        relevant_memories: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Prepare context for the LLM including system prompt, memories, and conversation.
        
        Args:
            user_message: Current user message
            relevant_memories: Relevant long-term memories
            
        Returns:
            List of message dictionaries for the LLM
        """
        messages = []
        
        # Add system prompt
        system_content = self.system_prompt
        
        # Add relevant long-term memories to system prompt
        if relevant_memories:
            memory_context = "\n\nRELEVANT MEMORIES:\n"
            for i, memory in enumerate(relevant_memories, 1):
                memory_context += f"{i}. [{memory['memory_type'].upper()}] {memory['content']}"
                if memory.get('context'):
                    memory_context += f" (Context: {memory['context']})"
                memory_context += f" [Similarity: {memory['similarity']:.2f}]\n"
            
            system_content += memory_context
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history (excluding the current message)
        conversation_history = self.short_term_memory.get_conversation_context(
            include_system=False
        )
        
        # Remove the last message if it's the current user message
        if (conversation_history and 
            conversation_history[-1]["role"] == "user" and 
            conversation_history[-1]["content"] == user_message):
            conversation_history = conversation_history[:-1]
        
        messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _generate_response(self, context: List[Dict[str, str]]) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            context: List of message dictionaries
            
        Returns:
            Generated response
        """
        try:
            # Convert to LangChain message format
            langchain_messages = []
            for msg in context:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = await self.llm.ainvoke(langchain_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def _analyze_for_memory_storage(self, user_message: str, response: str,
                                        relevant_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze the conversation to suggest what should be stored in long-term memory.
        
        Args:
            user_message: User's message
            response: Agent's response
            relevant_memories: Retrieved memories
            
        Returns:
            List of memory storage suggestions
        """
        suggestions = []
        
        # Simple heuristics for memory storage (can be enhanced with ML)
        
        # Check for user preferences
        preference_keywords = ["i like", "i prefer", "i don't like", "i hate", "my favorite", "i love", "i enjoy"]
        if any(keyword in user_message.lower() for keyword in preference_keywords):
            suggestions.append({
                'type': 'preference',
                'content': user_message,
                'reason': 'User expressed a preference',
                'importance': 0.8
            })
        
        # Check for personal information
        personal_keywords = ["my name is", "i am", "i work", "i live", "my job"]
        if any(keyword in user_message.lower() for keyword in personal_keywords):
            suggestions.append({
                'type': 'fact',
                'content': user_message,
                'reason': 'User shared personal information',
                'importance': 0.9
            })
        
        # Check for important facts or knowledge
        fact_keywords = ["remember that", "important", "don't forget", "note that"]
        if any(keyword in user_message.lower() for keyword in fact_keywords):
            suggestions.append({
                'type': 'fact',
                'content': user_message,
                'reason': 'User indicated this is important to remember',
                'importance': 0.8
            })
        
        return suggestions
    
    async def store_memory_suggestion(self, suggestion: Dict[str, Any]) -> str:
        """
        Store a memory suggestion in long-term memory.
        
        Args:
            suggestion: Memory suggestion dictionary
            
        Returns:
            Memory ID
        """
        return await self.long_term_memory.store_memory(
            content=suggestion['content'],
            memory_type=suggestion['type'],
            importance=suggestion['importance'],
            context=f"Stored from conversation - {suggestion['reason']}"
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about both memory systems."""
        return {
            'short_term': self.short_term_memory.get_stats(),
            'long_term': self.long_term_memory.get_stats()
        }
    
    def clear_short_term_memory(self) -> None:
        """Clear short-term memory."""
        self.short_term_memory.clear()
    
    def clear_long_term_memory(self) -> bool:
        """Clear long-term memory."""
        return self.long_term_memory.clear_all_memories()
