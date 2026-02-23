"""
SQLite database setup for Frisbot.
Simple, lightweight persistence for companion states.
"""

import sqlite3
import json
import time
from typing import Dict, Optional, List, Any
from contextlib import contextmanager
import os


class Database:
    """
    SQLite database manager for Frisbot.
    Handles companion states, conversations, and session management.
    """
    
    def __init__(self, db_path: str = "frisbot.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Companions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companions (
                    id TEXT PRIMARY KEY,
                    name TEXT DEFAULT 'Frisbot',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    cognitive_state TEXT NOT NULL,  -- JSON blob
                    metadata TEXT  -- JSON blob for additional data
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    state_at_start TEXT,  -- JSON snapshot
                    state_at_end TEXT,    -- JSON snapshot
                    FOREIGN KEY (companion_id) REFERENCES companions(id)
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    companion_id TEXT NOT NULL,
                    session_id TEXT,
                    role TEXT NOT NULL,  -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    cognitive_state TEXT,  -- JSON snapshot at time of message
                    message_analysis TEXT,  -- JSON from analyzer
                    created_at REAL NOT NULL,
                    FOREIGN KEY (companion_id) REFERENCES companions(id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Topic beliefs table (normalized from companion state)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topic_beliefs (
                    companion_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    mu REAL NOT NULL,
                    sigma REAL NOT NULL,
                    last_discussed REAL,
                    discussion_count INTEGER DEFAULT 0,
                    PRIMARY KEY (companion_id, topic),
                    FOREIGN KEY (companion_id) REFERENCES companions(id)
                )
            """)
            
            # Reflection logs table (for System 2 when implemented)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    triggered_by TEXT,
                    accumulated_surprise REAL,
                    belief_changes TEXT,  -- JSON
                    reflection_narrative TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (companion_id) REFERENCES companions(id)
                )
            """)
            
            # Indices for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_companion 
                ON messages(companion_id, created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_companion
                ON sessions(companion_id, started_at DESC)
            """)
    
    # === Companion Management ===
    
    def create_companion(
        self, 
        companion_id: str,
        name: str,
        cognitive_state: Dict
    ) -> bool:
        """Create a new companion."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO companions (id, name, created_at, updated_at, cognitive_state)
                VALUES (?, ?, ?, ?, ?)
            """, (
                companion_id,
                name,
                time.time(),
                time.time(),
                json.dumps(cognitive_state)
            ))
            return cursor.rowcount > 0
    
    def get_companion(self, companion_id: str) -> Optional[Dict]:
        """Get companion by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM companions WHERE id = ?
            """, (companion_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'cognitive_state': json.loads(row['cognitive_state']),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None
    
    def update_companion_state(
        self,
        companion_id: str,
        cognitive_state: Dict
    ) -> bool:
        """Update companion's cognitive state."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE companions 
                SET cognitive_state = ?, updated_at = ?
                WHERE id = ?
            """, (
                json.dumps(cognitive_state),
                time.time(),
                companion_id
            ))
            return cursor.rowcount > 0
    
    def list_companions(self) -> List[Dict]:
        """List all companions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, created_at, updated_at 
                FROM companions 
                ORDER BY updated_at DESC
            """)
            
            return [
                {
                    'id': row['id'],
                    'name': row['name'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
                for row in cursor.fetchall()
            ]
    
    # === Message Management ===
    
    def save_message(
        self,
        companion_id: str,
        role: str,
        content: str,
        cognitive_state: Optional[Dict] = None,
        message_analysis: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> int:
        """Save a message to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages 
                (companion_id, session_id, role, content, cognitive_state, message_analysis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                companion_id,
                session_id,
                role,
                content,
                json.dumps(cognitive_state) if cognitive_state else None,
                json.dumps(message_analysis) if message_analysis else None,
                time.time()
            ))
            return cursor.lastrowid
    
    def get_conversation_history(
        self,
        companion_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """Get recent conversation history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, created_at, message_analysis
                FROM messages
                WHERE companion_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (companion_id, limit))
            
            # Reverse to get chronological order
            messages = []
            for row in reversed(cursor.fetchall()):
                messages.append({
                    'role': row['role'],
                    'content': row['content'],
                    'created_at': row['created_at'],
                    'analysis': json.loads(row['message_analysis']) if row['message_analysis'] else None
                })
            
            return messages
    
    # === Session Management ===
    
    def create_session(
        self,
        session_id: str,
        companion_id: str,
        initial_state: Dict
    ) -> bool:
        """Create a new conversation session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, companion_id, started_at, state_at_start)
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                companion_id,
                time.time(),
                json.dumps(initial_state)
            ))
            return cursor.rowcount > 0
    
    def end_session(
        self,
        session_id: str,
        final_state: Dict
    ) -> bool:
        """End a conversation session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET ended_at = ?, state_at_end = ?
                WHERE id = ?
            """, (
                time.time(),
                json.dumps(final_state),
                session_id
            ))
            return cursor.rowcount > 0
    
    # === Topic Beliefs ===
    
    def update_topic_belief(
        self,
        companion_id: str,
        topic: str,
        mu: float,
        sigma: float
    ) -> bool:
        """Update or insert topic belief."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO topic_beliefs 
                (companion_id, topic, mu, sigma, last_discussed, discussion_count)
                VALUES (?, ?, ?, ?, ?, 
                    COALESCE((SELECT discussion_count + 1 FROM topic_beliefs 
                              WHERE companion_id = ? AND topic = ?), 1))
            """, (
                companion_id, topic, mu, sigma, time.time(),
                companion_id, topic
            ))
            return cursor.rowcount > 0
    
    def get_topic_beliefs(self, companion_id: str) -> Dict[str, Dict]:
        """Get all topic beliefs for a companion."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT topic, mu, sigma, last_discussed, discussion_count
                FROM topic_beliefs
                WHERE companion_id = ?
            """, (companion_id,))
            
            beliefs = {}
            for row in cursor.fetchall():
                beliefs[row['topic']] = {
                    'mu': row['mu'],
                    'sigma': row['sigma'],
                    'last_discussed': row['last_discussed'],
                    'discussion_count': row['discussion_count']
                }
            
            return beliefs
    
    # === Analytics ===
    
    def get_companion_stats(self, companion_id: str) -> Dict:
        """Get statistics for a companion."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Message count
            cursor.execute("""
                SELECT COUNT(*) as total_messages,
                       COUNT(DISTINCT session_id) as total_sessions
                FROM messages
                WHERE companion_id = ?
            """, (companion_id,))
            
            row = cursor.fetchone()
            
            # Topic count
            cursor.execute("""
                SELECT COUNT(*) as topic_count
                FROM topic_beliefs
                WHERE companion_id = ?
            """, (companion_id,))
            
            topic_row = cursor.fetchone()
            
            return {
                'total_messages': row['total_messages'],
                'total_sessions': row['total_sessions'],
                'topics_discussed': topic_row['topic_count']
            }