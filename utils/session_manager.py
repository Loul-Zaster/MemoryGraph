"""
User and Session Management for the Memory Agent System.
"""

import uuid
import time
import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class ChatSession:
    """Represents a chat session."""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        return cls(**data)


@dataclass
class User:
    """Represents a user."""
    user_id: str
    username: str
    created_at: float
    last_login: float
    total_sessions: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(**data)


class SessionManager:
    """
    Manages users and chat sessions with proper isolation.
    """
    
    def __init__(self, data_dir: str = "./session_data"):
        """Initialize session manager."""
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.json")
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self.users: Dict[str, User] = self._load_users()
        self.sessions: Dict[str, ChatSession] = self._load_sessions()
        
        logger.info(f"SessionManager initialized with {len(self.users)} users and {len(self.sessions)} sessions")
    
    def _load_users(self) -> Dict[str, User]:
        """Load users from file."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {uid: User.from_dict(user_data) for uid, user_data in data.items()}
            except Exception as e:
                logger.error(f"Error loading users: {e}")
        return {}
    
    def _load_sessions(self) -> Dict[str, ChatSession]:
        """Load sessions from file."""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {sid: ChatSession.from_dict(session_data) for sid, session_data in data.items()}
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
        return {}
    
    def _save_users(self):
        """Save users to file."""
        try:
            data = {uid: user.to_dict() for uid, user in self.users.items()}
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _save_sessions(self):
        """Save sessions to file."""
        try:
            data = {sid: session.to_dict() for sid, session in self.sessions.items()}
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def create_user(self, username: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user."""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        current_time = time.time()
        
        user = User(
            user_id=user_id,
            username=username,
            created_at=current_time,
            last_login=current_time,
            total_sessions=0,
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        self._save_users()
        
        logger.info(f"Created user: {username} ({user_id})")
        return user_id
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new chat session for a user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        current_time = time.time()
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        
        # Update user stats
        user = self.users[user_id]
        user.total_sessions += 1
        user.last_login = current_time
        
        self._save_sessions()
        self._save_users()
        
        logger.info(f"Created session: {session_id} for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """Update session last activity."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = time.time()
            self._save_sessions()
    
    def end_session(self, session_id: str) -> bool:
        """End a chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            logger.info(f"Ended session: {session_id}")
            return True
        return False
    
    def get_user_sessions(self, user_id: str, active_only: bool = False) -> List[ChatSession]:
        """Get all sessions for a user."""
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        
        if active_only:
            # Consider sessions active if last activity was within 1 hour
            cutoff_time = time.time() - 3600
            user_sessions = [s for s in user_sessions if s.last_activity > cutoff_time]
        
        return sorted(user_sessions, key=lambda x: x.last_activity, reverse=True)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive sessions."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        old_sessions = [sid for sid, session in self.sessions.items() 
                       if session.last_activity < cutoff_time]
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        if old_sessions:
            self._save_sessions()
            logger.info(f"Cleaned up {len(old_sessions)} old sessions")
        
        return len(old_sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        active_sessions = len([s for s in self.sessions.values() 
                              if s.last_activity > time.time() - 3600])
        
        return {
            "total_users": len(self.users),
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "data_directory": self.data_dir
        }
    
    def generate_collection_name(self, user_id: str, session_id: str) -> str:
        """Generate unique collection name for ChromaDB."""
        return f"user_{user_id}_session_{session_id}"
    
    def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all their sessions."""
        if user_id not in self.users:
            return False
        
        # Delete all user sessions
        user_sessions = [sid for sid, session in self.sessions.items() 
                        if session.user_id == user_id]
        for session_id in user_sessions:
            del self.sessions[session_id]
        
        # Delete user
        del self.users[user_id]
        
        self._save_users()
        self._save_sessions()
        
        logger.info(f"Deleted user {user_id} and {len(user_sessions)} sessions")
        return True
