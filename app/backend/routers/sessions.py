"""
routers/sessions.py
Manages chat sessions for each user (used by sidebar and chat history).
Now integrated with MemoryAgent to delete/clear session-specific memories safely.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
import uuid
from app.agents.llm_memory import MemoryAgent

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# ----------------------------- Memory integration -----------------------------
memory_agent = MemoryAgent(persist_dir="./chroma_memory")

# ----------------------------- In-memory store (can be replaced by DB) -----------------------------
_sessions: Dict[str, Dict] = {}


# ----------------------------- Schemas -----------------------------
class NewSessionRequest(BaseModel):
    user_id: str
    title: str = "New Chat"


class UpdateSessionRequest(BaseModel):
    message: str


# ----------------------------- Endpoints -----------------------------
@router.get("")
def list_sessions(user_id: Optional[str] = Query(None)) -> List[Dict]:
    """
     Get all active chat sessions (for sidebar)
    Optionally filtered by user_id
    """
    sessions = list(_sessions.values())
    if user_id:
        sessions = [s for s in sessions if s["user_id"] == user_id]

    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


@router.post("")
def create_session(req: NewSessionRequest):
    """
     Create a new chat session for a user
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    _sessions[session_id] = {
        "session_id": session_id,
        "user_id": req.user_id,
        "title": req.title,
        "last_message": "",
        "updated_at": now,
    }

    return _sessions[session_id]


@router.get("/{session_id}")
def get_session(session_id: str):
    """
     Get session details
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.put("/{session_id}")
def update_session(session_id: str, req: UpdateSessionRequest):
    """
     Update session last_message and timestamp (called after each chat message)
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["last_message"] = req.message
    session["updated_at"] = datetime.utcnow().isoformat()
    return {"status": "updated", "session_id": session_id}


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """
     Delete a session and its associated memory (session-specific)
    """
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        memory_agent.delete_memory(user_id=session["user_id"], session_id=session_id)
    except Exception as e:
        # Log but continue to return deletion of in-memory session
        logger = memory_agent.__class__.__module__  # placeholder, avoid crash if logging not available
        # Re-raise as HTTPException
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "deleted", "session_id": session_id}


@router.post("/{session_id}/clear")
def clear_session_memory(session_id: str):
    """
     Clear memory for this session without deleting the session itself.
    Equivalent to "New Chat" button.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        memory_agent.delete_memory(user_id=session["user_id"], session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session["last_message"] = ""
    session["updated_at"] = datetime.utcnow().isoformat()
    return {"status": "cleared", "session_id": session_id}
