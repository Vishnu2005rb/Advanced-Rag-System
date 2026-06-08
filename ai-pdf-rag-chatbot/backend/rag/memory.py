from typing import List, Dict
from rag.database import get_chat_history, add_chat_message, clear_chat_history

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieves the message history for a given session ID from SQLite."""
    return get_chat_history(session_id)

def add_message(session_id: str, role: str, text: str):
    """Appends a new message to the history of a session ID in SQLite."""
    add_chat_message(session_id, role, text)

def format_history(session_id: str) -> str:
    """Formats the history array into a prompt-friendly string."""
    history = get_history(session_id)
    if not history:
        return "No prior conversation history."
        
    formatted = ""
    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['text']}\n"
    return formatted

def clear_history(session_id: str):
    """Resets memory for a specific session ID in SQLite."""
    clear_chat_history(session_id)
