import os
import sqlite3
import tempfile
import time
from typing import List, Dict, Optional

# Use writable directory (defaulting to system temp, but customizable via DATA_DIR environment variable)
DATA_DIR = os.getenv("DATA_DIR", tempfile.gettempdir())
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "registry.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes tables for document registry and chat memory."""
    with get_connection() as conn:
        # Document registry table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_registry (
                filename TEXT PRIMARY KEY,
                size TEXT,
                pages INTEGER,
                chunks INTEGER,
                embedding_model TEXT,
                vector_count INTEGER,
                status TEXT,
                timestamp REAL
            )
        """)
        # Chat memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                text TEXT,
                timestamp REAL
            )
        """)
        conn.commit()

# --- Registry Helper Functions ---

def save_document(doc_stats: Dict):
    """Saves or updates document metadata in the registry."""
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO document_registry 
            (filename, size, pages, chunks, embedding_model, vector_count, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_stats["filename"],
            doc_stats["size"],
            doc_stats["pages"],
            doc_stats["chunks"],
            doc_stats["embedding_model"],
            doc_stats["vector_count"],
            doc_stats["status"],
            doc_stats["timestamp"]
        ))
        conn.commit()

def get_all_documents() -> List[Dict]:
    """Retrieves all registered documents."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM document_registry ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_document_by_filename(filename: str) -> Optional[Dict]:
    """Retrieves a single document by filename."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM document_registry WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_document(filename: str):
    """Deletes a document from the registry by filename."""
    with get_connection() as conn:
        conn.execute("DELETE FROM document_registry WHERE filename = ?", (filename,))
        conn.commit()

def clear_registry():
    """Clears all documents from the registry."""
    with get_connection() as conn:
        conn.execute("DELETE FROM document_registry")
        conn.commit()

# --- Chat Memory Helper Functions ---

def add_chat_message(session_id: str, role: str, text: str):
    """Appends a chat message and caps the history to the last 20 messages."""
    with get_connection() as conn:
        # Insert new message
        conn.execute("""
            INSERT INTO chat_history (session_id, role, text, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, text, time.time()))
        
        # Keep only the last 20 messages for this session
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM chat_history 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 20
        """, (session_id,))
        ids_to_keep = [row[0] for row in cursor.fetchall()]
        
        if ids_to_keep:
            # Delete any message that is NOT in the latest 20
            placeholders = ",".join("?" for _ in ids_to_keep)
            conn.execute(f"""
                DELETE FROM chat_history 
                WHERE session_id = ? AND id NOT IN ({placeholders})
            """, (session_id, *ids_to_keep))
        
        conn.commit()

def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieves chat history for a session ID, ordered oldest to newest."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, text FROM chat_history 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        rows = cursor.fetchall()
        return [{"role": row[0], "text": row[1]} for row in rows]

def clear_chat_history(session_id: str):
    """Resets memory for a specific session ID."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
