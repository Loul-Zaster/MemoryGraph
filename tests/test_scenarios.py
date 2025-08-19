"""
Comprehensive test scenarios for memory functionality.
"""

import asyncio
import time
from typing import Dict, Any, List
from loguru import logger
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph_builder import create_memory_workflow
from config import Config


class TestScenarios:
    """Test scenarios for the memory agent system."""
    
    def __init__(self):
        """Initialize test scenarios."""
        self.workflow = None
        self.test_results = []
    
    async def setup(self):
        """Set up the test environment."""
        print("🔧 Setting up test environment...")
        try:
            self.workflow = create_memory_workflow()
            print("✅ Test environment ready!")
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    async def teardown(self):
        """Clean up after tests."""
        print("🧹 Cleaning up test environment...")
        if self.workflow:
            # Clear memories for clean tests
            self.workflow.get_agent().clear_short_term_memory()
            self.workflow.get_agent().clear_long_term_memory()
        print("✅ Cleanup complete!")
    
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        })
    
    async def test_basic_memory_storage(self):
        """Test basic memory storage and retrieval."""
        print("\n🧪 Testing basic memory storage...")

        try:
            # Ensure workflow is set up
            if not self.workflow:
                await self.setup()

            # Test storing a user preference
            result1 = await self.workflow.run("I love pizza, especially pepperoni pizza!")
            
            # Check if memory suggestions were generated
            has_suggestions = len(result1.get('memory_suggestions', [])) > 0
            self.log_test_result(
                "Basic memory suggestion generation",
                has_suggestions,
                f"Generated {len(result1.get('memory_suggestions', []))} suggestions"
            )
            
            # Test retrieving the memory
            result2 = await self.workflow.run("What kind of food do I like?")
            
            # Check if relevant memories were retrieved
            has_memories = len(result2.get('relevant_memories', [])) > 0
            self.log_test_result(
                "Basic memory retrieval",
                has_memories,
                f"Retrieved {len(result2.get('relevant_memories', []))} memories"
            )
            
            # Check if the response mentions pizza
            mentions_pizza = "pizza" in result2['response'].lower()
            self.log_test_result(
                "Memory-informed response",
                mentions_pizza,
                "Response includes remembered preference"
            )
            
        except Exception as e:
            self.log_test_result("Basic memory storage", False, f"Error: {e}")
    
    async def test_short_term_memory(self):
        """Test short-term memory functionality."""
        print("\n🧪 Testing short-term memory...")
        
        try:
            # Clear previous context
            self.workflow.get_agent().clear_short_term_memory()
            
            # First message
            await self.workflow.run("My name is Alice and I'm a software engineer.")
            
            # Second message referring to previous context
            result = await self.workflow.run("What did I just tell you about myself?")
            
            # Check if the response includes the information from the previous message
            includes_name = "alice" in result['response'].lower()
            includes_job = "software" in result['response'].lower() or "engineer" in result['response'].lower()
            
            self.log_test_result(
                "Short-term memory recall",
                includes_name and includes_job,
                "Agent remembered name and profession from previous message"
            )
            
            # Test conversation context
            context = self.workflow.get_agent().short_term_memory.get_conversation_context()
            has_context = len(context) >= 4  # 2 user messages + 2 assistant responses
            
            self.log_test_result(
                "Conversation context maintenance",
                has_context,
                f"Maintained {len(context)} messages in context"
            )
            
        except Exception as e:
            self.log_test_result("Short-term memory", False, f"Error: {e}")
    
    async def test_long_term_memory_persistence(self):
        """Test long-term memory persistence across sessions."""
        print("\n🧪 Testing long-term memory persistence...")
        
        try:
            # Store some important information
            result1 = await self.workflow.run("Remember that my birthday is on December 25th and I'm allergic to peanuts.")
            
            # Clear short-term memory to simulate a new session
            self.workflow.get_agent().clear_short_term_memory()
            
            # Try to retrieve the information in a "new session"
            result2 = await self.workflow.run("When is my birthday?")
            
            # Check if the birthday information was retrieved
            mentions_birthday = "december" in result2['response'].lower() or "25" in result2['response']
            has_memories = len(result2.get('relevant_memories', [])) > 0
            
            self.log_test_result(
                "Long-term memory persistence",
                mentions_birthday and has_memories,
                "Retrieved birthday information after clearing short-term memory"
            )
            
            # Test allergy information
            result3 = await self.workflow.run("What foods should I avoid?")
            mentions_peanuts = "peanut" in result3['response'].lower()
            
            self.log_test_result(
                "Multiple long-term memories",
                mentions_peanuts,
                "Retrieved allergy information"
            )
            
        except Exception as e:
            self.log_test_result("Long-term memory persistence", False, f"Error: {e}")
    
    async def test_memory_types(self):
        """Test different types of memory storage."""
        print("\n🧪 Testing different memory types...")
        
        try:
            # Test preference memory
            await self.workflow.run("I prefer working in the morning and I don't like loud music.")
            
            # Test factual memory
            await self.workflow.run("Remember that the capital of France is Paris.")
            
            # Test personal information
            await self.workflow.run("I work at TechCorp as a data scientist.")
            
            # Query for preferences
            result1 = await self.workflow.run("What are my work preferences?")
            mentions_morning = "morning" in result1['response'].lower()
            
            # Query for facts
            result2 = await self.workflow.run("What's the capital of France?")
            mentions_paris = "paris" in result2['response'].lower()
            
            # Query for personal info
            result3 = await self.workflow.run("Where do I work?")
            mentions_techcorp = "techcorp" in result3['response'].lower()
            
            self.log_test_result(
                "Preference memory",
                mentions_morning,
                "Retrieved work preference"
            )
            
            self.log_test_result(
                "Factual memory",
                mentions_paris,
                "Retrieved factual information"
            )
            
            self.log_test_result(
                "Personal information memory",
                mentions_techcorp,
                "Retrieved personal information"
            )
            
        except Exception as e:
            self.log_test_result("Memory types", False, f"Error: {e}")
    
    async def test_memory_search_accuracy(self):
        """Test the accuracy of memory search and retrieval."""
        print("\n🧪 Testing memory search accuracy...")
        
        try:
            # Store multiple different pieces of information
            await self.workflow.run("I love Italian food, especially pasta and pizza.")
            await self.workflow.run("My favorite color is blue and I enjoy reading science fiction books.")
            await self.workflow.run("I have a cat named Whiskers and a dog named Max.")
            
            # Test specific queries
            result1 = await self.workflow.run("What pets do I have?")
            mentions_pets = ("cat" in result1['response'].lower() or "whiskers" in result1['response'].lower() or 
                           "dog" in result1['response'].lower() or "max" in result1['response'].lower())
            
            result2 = await self.workflow.run("What's my favorite cuisine?")
            mentions_italian = "italian" in result2['response'].lower() or "pasta" in result2['response'].lower()
            
            result3 = await self.workflow.run("What books do I like?")
            mentions_scifi = ("science fiction" in result3['response'].lower() or 
                            "sci-fi" in result3['response'].lower() or "scifi" in result3['response'].lower())
            
            self.log_test_result(
                "Pet information retrieval",
                mentions_pets,
                "Correctly retrieved pet information"
            )
            
            self.log_test_result(
                "Food preference retrieval",
                mentions_italian,
                "Correctly retrieved food preferences"
            )
            
            self.log_test_result(
                "Book preference retrieval",
                mentions_scifi,
                "Correctly retrieved reading preferences"
            )
            
        except Exception as e:
            self.log_test_result("Memory search accuracy", False, f"Error: {e}")
    
    async def test_memory_statistics(self):
        """Test memory statistics and management."""
        print("\n🧪 Testing memory statistics...")
        
        try:
            # Add some messages and memories
            await self.workflow.run("Hello, I'm testing the memory system.")
            await self.workflow.run("I like chocolate ice cream.")
            await self.workflow.run("My favorite movie is The Matrix.")
            
            # Get statistics
            stats = self.workflow.get_memory_stats()
            
            has_short_term = stats.get('short_term', {}).get('total_messages', 0) > 0
            has_long_term = stats.get('long_term', {}).get('total_memories', 0) >= 0
            
            self.log_test_result(
                "Memory statistics generation",
                has_short_term and 'long_term' in stats,
                f"Short-term: {stats.get('short_term', {}).get('total_messages', 0)} messages, "
                f"Long-term: {stats.get('long_term', {}).get('total_memories', 0)} memories"
            )
            
            # Test memory clearing
            self.workflow.get_agent().clear_short_term_memory()
            stats_after_clear = self.workflow.get_memory_stats()
            
            cleared_successfully = stats_after_clear.get('short_term', {}).get('total_messages', 0) == 0
            
            self.log_test_result(
                "Short-term memory clearing",
                cleared_successfully,
                "Successfully cleared short-term memory"
            )
            
        except Exception as e:
            self.log_test_result("Memory statistics", False, f"Error: {e}")
    
    async def test_all_scenarios(self):
        """Run all test scenarios."""
        print("🚀 Running all test scenarios...")
        print("=" * 60)
        
        if not await self.setup():
            return
        
        try:
            # Run all test methods
            await self.test_basic_memory_storage()
            await self.test_short_term_memory()
            await self.test_long_term_memory_persistence()
            await self.test_memory_types()
            await self.test_memory_search_accuracy()
            await self.test_memory_statistics()
            
        finally:
            await self.teardown()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        if failed_tests > 0:
            print("\n❌ Failed tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['details']}")
        
        print("=" * 60)
