from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime
import logging
import asyncio
from pathlib import Path

# Import your RAGAssistant
from app.generation.engine import RAGAssistant

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None
    use_subqueries: Optional[bool] = True  # optional for splitting complex queries

# Initialize RAGAssistant once (singleton)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

assistant = RAGAssistant(config_file=CONFIG_FILE, use_local_llm=False)

# In-memory session store
_sessions = {}

@router.post("")
async def chat(req: ChatRequest):
    """
    Handle a chat message using RAGAssistant and return response.
    """
    user_id = req.user_id
    message = req.message
    session_id = req.session_id or str(uuid.uuid4())
    use_subqueries = req.use_subqueries

    now = datetime.utcnow().isoformat()
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "title": "Chat",
            "last_message": "",
            "updated_at": now,
        }

    logger.info(f"CHAT request user={user_id} session={session_id} message={message[:120]}")

    try:
        # Use asyncio.to_thread because generate_answer is blocking
        reply = await asyncio.to_thread(
            assistant.generate_answer,
            message
        )
    except Exception as e:
        logger.exception("Error generating RAG response for user=%s session=%s: %s", user_id, session_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    # Update session metadata
    _sessions[session_id]["last_message"] = message
    _sessions[session_id]["updated_at"] = datetime.utcnow().isoformat()

    return {
        "session_id": session_id,
        "reply": reply,
        "retrieved_docs": None  # optionally, you can modify assistant to return docs
    }
