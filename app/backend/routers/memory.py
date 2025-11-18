"""
routers/memory.py
FastAPI endpoints for managing user & session-specific memories using MemoryAgent.
"""

from fastapi import APIRouter, HTTPException, Query
from app.agents.llm_memory import MemoryAgent
from typing import Optional

router = APIRouter(prefix="/api/memory", tags=["memory"])
memory_agent = MemoryAgent(persist_dir="./chroma_memory")


@router.get("/{user_id}")
def get_user_memory(user_id: str, session_id: Optional[str] = Query(None)):
    """
     Get all stored memories for a specific user (or for a session if session_id provided).
    """
    try:
        data = memory_agent.export_user_memory(user_id, session_id)
        return {"user_id": user_id, "session_id": session_id, "count": len(data), "memories": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
def delete_user_memory(user_id: str, session_id: Optional[str] = Query(None)):
    """
    Delete memories for a user or a specific session.
    """
    try:
        memory_agent.delete_memory(user_id=user_id, session_id=session_id)
        return {"status": "deleted", "user_id": user_id, "session_id": session_id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
def clear_all_memory():
    """
     Clear the entire memory collection (admin/debug).
    CAUTION: This deletes everything.
    """
    try:
        # Use the underlying collection delete but keep it explicit
        memory_agent.collection.delete(where={})
        return {"status": "cleared_all"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
