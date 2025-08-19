"""
Main entry point for the Multi-Agent Memory Test System.
"""

import asyncio
import sys
import os
import time
from typing import Optional
from loguru import logger
from config import Config
from agents.graph_builder import create_memory_workflow
from utils.session_manager import SessionManager


def setup_logging():
    """Set up logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format=Config.LOG_FORMAT,
        level=Config.LOG_LEVEL,
        colorize=True
    )
    logger.add(
        "logs/memory_agent.log",
        format=Config.LOG_FORMAT,
        level=Config.LOG_LEVEL,
        rotation="1 day",
        retention="7 days"
    )


async def interactive_chat():
    """Run an interactive chat session with the memory agent."""
    print("🤖 Memory Agent Chat Interface")
    print("=" * 50)

    # Initialize session manager
    session_manager = SessionManager()

    # User login/creation
    username = input("👤 Enter your username: ").strip()
    if not username:
        username = "anonymous"

    user = session_manager.find_user_by_username(username)
    if not user:
        user_id = session_manager.create_user(username)
        print(f"✅ Created new user: {username} ({user_id})")
    else:
        user_id = user.user_id
        print(f"👋 Welcome back, {username}! ({user_id})")

    # Create new session
    session_id = session_manager.create_session(user_id)
    print(f"🆔 Session ID: {session_id}")

    print("=" * 50)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'stats' to see memory statistics")
    print("Type 'clear' to clear short-term memory")
    print("Type 'help' for more commands")
    print("=" * 50)

    # Create the workflow with user/session context
    try:
        workflow = create_memory_workflow(user_id=user_id, session_id=session_id)
        print("✅ Memory agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting!")
                break
            
            elif user_input.lower() == 'stats':
                stats = workflow.get_memory_stats()
                session_stats = session_manager.get_stats()
                print("\n📊 Memory Statistics:")
                print(f"Short-term messages: {stats['short_term']['total_messages']}")
                print(f"Long-term memories: {stats['long_term']['total_memories']}")
                print(f"User ID: {user_id}")
                print(f"Session ID: {session_id}")
                print(f"Total users: {session_stats['total_users']}")
                print(f"Active sessions: {session_stats['active_sessions']}")
                continue
            
            elif user_input.lower() == 'clear':
                workflow.get_agent().clear_short_term_memory()
                print("🧹 Short-term memory cleared!")
                continue
            
            elif user_input.lower() == 'help':
                print("\n🆘 Available Commands:")
                print("- 'stats': Show memory statistics")
                print("- 'clear': Clear short-term memory")
                print("- 'sessions': Show your session history")
                print("- 'users': Show system users (admin)")
                print("- 'cleanup': Clean up old sessions")
                print("- 'quit'/'exit'/'bye': End conversation")
                print("- 'help': Show this help message")
                continue

            elif user_input.lower() == 'sessions':
                user_sessions = session_manager.get_user_sessions(user_id)
                print(f"\n📋 Your Sessions ({len(user_sessions)} total):")
                for i, session in enumerate(user_sessions[:5], 1):  # Show last 5
                    status = "🟢 Current" if session.session_id == session_id else "⚪ Past"
                    print(f"{i}. {status} {session.session_id} - {time.strftime('%Y-%m-%d %H:%M', time.localtime(session.created_at))}")
                continue

            elif user_input.lower() == 'users':
                users = session_manager.list_users()
                print(f"\n👥 System Users ({len(users)} total):")
                for i, user in enumerate(users[:10], 1):  # Show first 10
                    print(f"{i}. {user.username} ({user.user_id}) - {user.total_sessions} sessions")
                continue

            elif user_input.lower() == 'cleanup':
                cleaned = session_manager.cleanup_old_sessions()
                print(f"🧹 Cleaned up {cleaned} old sessions")
                continue
            
            # Update session activity
            session_manager.update_session_activity(session_id)

            # Process the message
            print("🤔 Thinking...")
            result = await workflow.run(user_input)

            # Display response
            print(f"\n🤖 Agent: {result['response']}")

            # Show memory information if relevant
            if result['relevant_memories']:
                print(f"\n💭 Used {len(result['relevant_memories'])} relevant memories")

            if result['memory_suggestions']:
                print(f"💾 Suggested {len(result['memory_suggestions'])} items for long-term storage")

            if result['stored_memory_ids']:
                print(f"✅ Stored {len(result['stored_memory_ids'])} new memories")

            if result.get('error'):
                print(f"⚠️ Warning: {result['error']}")

            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error in interactive chat: {e}")

    # End session
    session_manager.end_session(session_id)
    print(f"🔚 Session {session_id} ended")


async def run_test_scenario(scenario_name: str):
    """Run a specific test scenario."""
    print(f"🧪 Running test scenario: {scenario_name}")
    
    # Import test scenarios
    try:
        from tests.test_scenarios import TestScenarios
        test_runner = TestScenarios()
        
        if hasattr(test_runner, scenario_name):
            await getattr(test_runner, scenario_name)()
        else:
            print(f"❌ Test scenario '{scenario_name}' not found")
            print("Available scenarios:")
            for attr in dir(test_runner):
                if attr.startswith('test_') and callable(getattr(test_runner, attr)):
                    print(f"  - {attr}")
    
    except ImportError as e:
        print(f"❌ Error importing test scenarios: {e}")
    except Exception as e:
        print(f"❌ Error running test scenario: {e}")


async def main():
    """Main function."""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging
    setup_logging()
    
    # Validate configuration
    if not Config.validate_config():
        print("❌ Configuration validation failed. Please check your settings.")
        return
    
    print("🚀 Multi-Agent Memory Test System")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "chat":
            await interactive_chat()
        elif command == "test":
            if len(sys.argv) > 2:
                scenario = sys.argv[2]
                await run_test_scenario(scenario)
            else:
                await run_test_scenario("test_all_scenarios")
        elif command == "help":
            print_help()
        else:
            print(f"❌ Unknown command: {command}")
            print_help()
    else:
        # Default to interactive chat
        await interactive_chat()


def print_help():
    """Print help information."""
    print("Usage: python main.py [command] [options]")
    print("\nCommands:")
    print("  chat                 Start interactive chat (default)")
    print("  test [scenario]      Run test scenarios")
    print("  help                 Show this help message")
    print("\nExamples:")
    print("  python main.py                          # Start interactive chat")
    print("  python main.py chat                     # Start interactive chat")
    print("  python main.py test                     # Run all test scenarios")
    print("  python main.py test test_basic_memory   # Run specific test")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
