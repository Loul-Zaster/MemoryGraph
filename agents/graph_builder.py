"""
LangGraph workflow definition for the memory agent system.
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage
from loguru import logger
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.memory_agent import MemoryAgent
from config import Config


class AgentState(TypedDict):
    """State definition for the agent workflow."""
    messages: List[BaseMessage]
    user_input: str
    agent_response: str
    relevant_memories: List[Dict[str, Any]]
    memory_suggestions: List[Dict[str, Any]]
    should_store_memory: bool
    stored_memory_ids: List[str]
    error: Optional[str]


class MemoryAgentWorkflow:
    """
    LangGraph workflow for the memory-enabled agent.
    
    This workflow orchestrates the interaction between user input,
    memory retrieval, response generation, and memory storage.
    """
    
    def __init__(self, agent: Optional[MemoryAgent] = None):
        """
        Initialize the workflow.
        
        Args:
            agent: Optional MemoryAgent instance
        """
        self.agent = agent or MemoryAgent()
        self.graph = self._build_graph()
        
        logger.info("Initialized MemoryAgentWorkflow")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input_node)
        workflow.add_node("retrieve_memories", self._retrieve_memories_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("analyze_memory", self._analyze_memory_node)
        workflow.add_node("store_memory", self._store_memory_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define the workflow edges
        workflow.set_entry_point("process_input")
        
        workflow.add_edge("process_input", "retrieve_memories")
        workflow.add_edge("retrieve_memories", "generate_response")
        workflow.add_edge("generate_response", "analyze_memory")
        
        # Conditional edge for memory storage
        workflow.add_conditional_edges(
            "analyze_memory",
            self._should_store_memory,
            {
                "store": "store_memory",
                "skip": "finalize"
            }
        )
        
        workflow.add_edge("store_memory", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _process_input_node(self, state: AgentState) -> AgentState:
        """Process the user input."""
        try:
            logger.debug("Processing user input")
            
            # Add user message to short-term memory
            self.agent.short_term_memory.add_user_message(state["user_input"])
            
            # Update state
            state["error"] = None
            
            return state
            
        except Exception as e:
            logger.error(f"Error in process_input_node: {e}")
            state["error"] = str(e)
            return state
    
    async def _retrieve_memories_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant long-term memories."""
        try:
            logger.debug("Retrieving relevant memories")
            
            # Retrieve relevant memories
            relevant_memories = await self.agent.long_term_memory.retrieve_memories(
                query=state["user_input"],
                max_results=Config.MAX_LONG_TERM_MEMORIES
            )
            
            state["relevant_memories"] = relevant_memories
            
            logger.debug(f"Retrieved {len(relevant_memories)} relevant memories")
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve_memories_node: {e}")
            state["error"] = str(e)
            state["relevant_memories"] = []
            return state
    
    async def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate the agent response."""
        try:
            logger.debug("Generating response")
            
            # Prepare context
            context = self.agent._prepare_context(
                state["user_input"], 
                state["relevant_memories"]
            )
            
            # Generate response
            response = await self.agent._generate_response(context)
            
            # Add response to short-term memory
            self.agent.short_term_memory.add_assistant_message(response)
            
            state["agent_response"] = response
            
            logger.debug("Response generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in generate_response_node: {e}")
            state["error"] = str(e)
            state["agent_response"] = "I apologize, but I encountered an error generating a response."
            return state
    
    async def _analyze_memory_node(self, state: AgentState) -> AgentState:
        """Analyze if anything should be stored in memory."""
        try:
            logger.debug("Analyzing for memory storage")
            
            # Analyze for memory storage suggestions
            memory_suggestions = await self.agent._analyze_for_memory_storage(
                state["user_input"],
                state["agent_response"],
                state["relevant_memories"]
            )
            
            state["memory_suggestions"] = memory_suggestions
            state["should_store_memory"] = len(memory_suggestions) > 0
            
            logger.debug(f"Found {len(memory_suggestions)} memory suggestions")
            return state
            
        except Exception as e:
            logger.error(f"Error in analyze_memory_node: {e}")
            state["error"] = str(e)
            state["memory_suggestions"] = []
            state["should_store_memory"] = False
            return state
    
    def _should_store_memory(self, state: AgentState) -> str:
        """Determine if memory should be stored."""
        return "store" if state.get("should_store_memory", False) else "skip"
    
    async def _store_memory_node(self, state: AgentState) -> AgentState:
        """Store suggested memories."""
        try:
            logger.debug("Storing memories")
            
            stored_ids = []
            
            for suggestion in state["memory_suggestions"]:
                try:
                    memory_id = await self.agent.store_memory_suggestion(suggestion)
                    stored_ids.append(memory_id)
                    logger.debug(f"Stored memory: {suggestion['type']} - {suggestion['content'][:50]}...")
                except Exception as e:
                    logger.error(f"Error storing memory suggestion: {e}")
            
            state["stored_memory_ids"] = stored_ids
            
            logger.debug(f"Stored {len(stored_ids)} memories")
            return state
            
        except Exception as e:
            logger.error(f"Error in store_memory_node: {e}")
            state["error"] = str(e)
            state["stored_memory_ids"] = []
            return state
    
    async def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the workflow."""
        logger.debug("Finalizing workflow")
        return state
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        Run the complete workflow for a user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dictionary containing the workflow results
        """
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[],
                user_input=user_input,
                agent_response="",
                relevant_memories=[],
                memory_suggestions=[],
                should_store_memory=False,
                stored_memory_ids=[],
                error=None
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return results
            return {
                'response': final_state['agent_response'],
                'relevant_memories': final_state['relevant_memories'],
                'memory_suggestions': final_state['memory_suggestions'],
                'stored_memory_ids': final_state['stored_memory_ids'],
                'error': final_state.get('error'),
                'short_term_context': self.agent.short_term_memory.get_conversation_context()
            }
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            return {
                'response': "I apologize, but I encountered an error processing your request.",
                'error': str(e),
                'relevant_memories': [],
                'memory_suggestions': [],
                'stored_memory_ids': [],
                'short_term_context': []
            }
    
    def get_agent(self) -> MemoryAgent:
        """Get the underlying agent."""
        return self.agent
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.agent.get_memory_stats()


# Utility function to create a workflow
def create_memory_workflow(api_key: Optional[str] = None,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> MemoryAgentWorkflow:
    """
    Create a memory agent workflow.

    Args:
        api_key: Optional OpenAI API key
        user_id: User ID for memory isolation
        session_id: Session ID for memory isolation

    Returns:
        MemoryAgentWorkflow instance
    """
    agent = MemoryAgent(api_key=api_key, user_id=user_id, session_id=session_id)
    return MemoryAgentWorkflow(agent)
