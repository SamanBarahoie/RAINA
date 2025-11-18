"""
agents/llm_memory.py

MemoryAgent manages user & session-specific memory using ChromaDB + Ollama embeddings.
It provides safe helpers for querying and deleting memory using Chroma's expected 'where' format.
"""

import chromadb
import uuid
import datetime
import json
import logging
import re
from typing import List, Dict, Any, Optional
import ollama  # Ollama embeddings engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MemoryAgent:
    """Memory system for multi-session user chat using Chroma + Ollama embeddings."""

    def __init__(self, persist_dir: str = "./chroma_memory", embedding_model_name: str = "nomic-embed-text"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="user_memories")
        self.embedding_model_name = embedding_model_name
        logger.info(f"MemoryAgent initialized with model: {embedding_model_name}")

    # ----------------------------- Helper for where clause -----------------------------
    def _build_where_clause(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a Chroma 'where' filter:
        - both user_id+session_id -> {"$and": [{"user_id": ...}, {"session_id": ...}]}
        - single -> {"user_id": ...} or {"session_id": ...}
        - none -> {}
        """
        if user_id and session_id:
            return {"$and": [{"user_id": user_id}, {"session_id": session_id}]}
        if user_id:
            return {"user_id": user_id}
        if session_id:
            return {"session_id": session_id}
        return {}

    # ----------------------------- Store Memory -----------------------------
    def store_memory(self, user_id: str, session_id: str, text: str) -> str:
        """Summarize and store chat message in memory with metadata (user_id + session_id)."""
        if not text or not text.strip():
            return ""

        summary = self._summarize_text(text)
        emb = self._embed(summary)
        mem_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()

        self.collection.add(
            documents=[summary],
            embeddings=[emb],
            metadatas=[{
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": timestamp
            }],
            ids=[mem_id],
        )

        logger.info(f"Stored memory (id={mem_id}) user={user_id} session={session_id}")
        return mem_id

    # ----------------------------- Retrieve Memory -----------------------------
    def retrieve_memory(self, user_id: Optional[str], session_id: Optional[str], query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the most relevant memories for a given user & session and query.
        Both user_id and session_id are optional (but recommended).
        """
        emb = self._embed(query)
        where_clause = self._build_where_clause(user_id=user_id, session_id=session_id)
        logger.debug(f"Querying memories where={where_clause} n_results={top_k}")
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            where=where_clause
        )
        docs = results.get("documents", [[]])[0]
        return docs or []

    # ----------------------------- Export user/session memory -----------------------------
    def export_user_memory(self, user_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all memories for a user or for a specific session."""
        where_clause = self._build_where_clause(user_id=user_id, session_id=session_id)
        logger.debug(f"Exporting memory where={where_clause}")
        results = self.collection.get(where=where_clause)
        data = []
        for doc, meta, mid in zip(results.get("documents", []), results.get("metadatas", []), results.get("ids", [])):
            data.append({
                "id": mid,
                "user_id": meta.get("user_id"),
                "session_id": meta.get("session_id"),
                "timestamp": meta.get("timestamp"),
                "summary": doc
            })
        return data

    def export_user_memory_json(
        self, user_id: str, session_id: Optional[str] = None, file_path: Optional[str] = None
    ) -> str:
        """Export user/session memory as JSON string and optionally save to disk."""
        data = self.export_user_memory(user_id, session_id)
        json_data = json.dumps(data, ensure_ascii=False, indent=2)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_data)
            logger.info(f"Exported memory → {file_path}")
        return json_data

    # ----------------------------- Delete Memory -----------------------------
    def delete_memory(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete memories matching the filters.
        If both user_id and session_id are provided, deletes only that session's memories.
        If none provided, raises ValueError to avoid accidental full deletion.
        """
        where_clause = self._build_where_clause(user_id=user_id, session_id=session_id)
        if not where_clause:
            raise ValueError("Refusing to delete with empty filters. Provide user_id and/or session_id.")
        logger.info(f"Deleting memories where={where_clause}")
        self.collection.delete(where=where_clause)
        return {"status": "deleted", "where": where_clause}

    # ----------------------------- Build memory context for prompt -----------------------------
    def build_context_for_prompt(self, user_id: Optional[str], session_id: Optional[str], query: str, max_memories: int = 5) -> str:
        """
        Build memory context block for LLM prompt using session-aware retrieval.
        If no memories found, returns a polite default message.
        """
        memories = self.retrieve_memory(user_id=user_id, session_id=session_id, query=query, top_k=max_memories)
        if not memories:
            return "هیچ پیش‌زمینه‌ای از مکالمات قبلی در این نشست وجود ندارد."

        context = "\n".join([f"- {m}" for m in memories])
        return f"سوابق گفتگو در این نشست:\n{context}\n"

    # ----------------------------- Internal Utilities -----------------------------
    def _summarize_text(self, text: str) -> str:
        """Simple rule-based summarization for Persian."""
        sentences = re.split(r"[.!؟?]\s*", text.strip())
        important = [s for s in sentences if len(s.split()) > 4 and any(v in s for v in ["است", "می‌شود", "دارد", "هست", "باید"])]
        summary = " ".join(important[:2]) if important else (sentences[0] if sentences else text)
        return summary.strip()

    def _embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama (sync call)."""
        response = ollama.embeddings(model=self.embedding_model_name, prompt=text)
        return response["embedding"]
