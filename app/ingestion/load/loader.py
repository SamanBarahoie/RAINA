import json
from pathlib import Path
from typing import List, Dict, Any, Optional


from ollama import Client as OllamaClient
from elasticsearch import Elasticsearch

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
class RAGStorage:
    """
    Stores summaries in ChromaDB (semantic search) and optionally full texts in Elasticsearch.
    Uses Ollama embeddings for vectorization. Works with ChromaDB HTTP mode.
    """

    def __init__(
        self,
        chroma_collection_name: str,
        es_index_name: Optional[str] = None,
        es_host: str = "http://elasticsearch:9200",
        ollama_model: str = "nomic-embed-text",
        chroma_path: str = None,   # Ignored in HTTP mode (kept for compatibility)
    ):
        # --- Ollama client ---
        self.ollama = OllamaClient(host="http://ollama:11434")
        self.model_name = ollama_model

        # --- ChromaDB HTTP Client ---
        self.chroma_client = chromadb.HttpClient(
            host="chroma",     # service name in docker-compose
            port=8000,
            settings=Settings(allow_reset=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=chroma_collection_name
        )

        # --- Elasticsearch (optional) ---
        self.es = None
        self.es_index = es_index_name
        if es_index_name:
            self.es = Elasticsearch(es_host)
            if not self.es.indices.exists(index=es_index_name):
                self.es.indices.create(index=es_index_name)

    @staticmethod
    def flatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata: convert None -> '', list/dict -> JSON string."""
        flat = {}
        for k, v in meta.items():
            if v is None:
                flat[k] = ""
            elif isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        return flat

    def store(self, data: List[Dict[str, Any]]):
        """
        Store summaries in ChromaDB and optionally full texts in Elasticsearch.
        Skip already stored chunk_ids to avoid duplicates.
        Generates a report after storing.
        """
        skipped = 0
        errors = 0
        stored = 0

        # Fetch existing IDs from ChromaDB to avoid duplicates
        try:
            existing_data = self.collection.get()
            existing_ids = set(existing_data["ids"]) if existing_data else set()
        except Exception:
            existing_ids = set()

        for item in data:
            chunk_id = f"{item['doc_id']}_{item['chunk_id']}"
            summary = item["metadata"].get("summary", "")
            full_text = item.get("chunk_text", "")
            metadata = self.flatten_metadata(item["metadata"])

            if chunk_id in existing_ids:
                skipped += 1
                continue

            try:
                # --- Compute embedding via Ollama ---
                response = self.ollama.embeddings(model=self.model_name, prompt=summary)
                embedding = response["embedding"]

                # --- Store in ChromaDB ---
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[metadata],
                )

                # --- Store full text in Elasticsearch (if enabled) ---
                if self.es:
                    if not self.es.exists(index=self.es_index, id=chunk_id):
                        self.es.index(
                            index=self.es_index,
                            id=chunk_id,
                            document={
                                "doc_id": item["doc_id"],
                                "chunk_id": item["chunk_id"],
                                "full_text": full_text,
                                "metadata": metadata,
                            },
                        )

                stored += 1

            except Exception as e:
                print(f"Error storing chunk_id {chunk_id}: {e}")
                errors += 1

        # --- Report ---
        print("\n=== Storage Report ===")
        print(f"Stored new chunks: {stored}")
        print(f"Skipped duplicates: {skipped}")
        print(f"Errors: {errors}")
        try:
            print(f"Total chunks in ChromaDB: {self.collection.count()}")
        except:
            pass

        if self.es:
            try:
                print(f"Total documents in Elasticsearch: {self.es.count(index=self.es_index)['count']}")
            except:
                pass

        print("======================\n")

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on ChromaDB and optionally attach full text."""
        response = self.ollama.embeddings(model=self.model_name, prompt=query)
        query_embedding = response["embedding"]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved_docs = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for i, doc_id in enumerate(ids):
            entry = {
                "doc_id": doc_id,
                "summary": docs[i],
                "metadata": metas[i],
            }
            if self.es:
                try:
                    es_doc = self.es.get(index=self.es_index, id=doc_id)
                    entry["full_text"] = es_doc["_source"].get("full_text", "")
                except Exception:
                    entry["full_text"] = ""
            else:
                entry["full_text"] = ""

            retrieved_docs.append(entry)

        return retrieved_docs

    def es_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search full texts in Elasticsearch (if enabled)."""
        if not self.es:
            return []

        body = {
            "size": top_k,
            "_source": ["doc_id", "chunk_id", "full_text", "metadata"],
            "query": {"match": {"full_text": {"query": query, "fuzziness": "AUTO"}}},
        }
        resp = self.es.search(index=self.es_index, body=body)
        hits = []
        for h in resp["hits"]["hits"]:
            source = h["_source"]
            hits.append({
                "doc_id": source["doc_id"],
                "chunk_id": source["chunk_id"],
                "full_text": source.get("full_text", ""),
                "summary": source["metadata"].get("summary", ""),
                "metadata": source["metadata"],
            })
        return hits


if __name__ == "__main__":
    storage = RAGStorage(
        chroma_collection_name="rag_sam_d",
        es_index_name="rag_data",
    )

    data = json.load(open("../../../downloads/cleaned_docs_llm.json", encoding="utf-8"))
    storage.store(data)

    query = "مراحل ثبت نام ترم تابستان"
    results = storage.semantic_search(query, top_k=3)
    print("=== Semantic Search Results ===")
    for r in results:
        print(r["doc_id"])
        print("Summary:", r["metadata"]["summary"])
        print("Full Text:", r["full_text"])
        print("------")
