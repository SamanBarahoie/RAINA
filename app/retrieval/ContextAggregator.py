from typing import List, Dict, Any, Optional
from app.ingestion.load.loader import RAGStorage
from app.agents.llm_api import LLM
from app.utils.utils import load_config


class QueryCollection:
    """
    Stores and retrieves semantically similar user queries in a separate Chroma collection.
    Used as a fallback when direct retrieval fails.
    """

    def __init__(self, chroma_path: str, collection_name: str = "query_collection"):
        # Create a Chroma-only RAGStorage (no Elasticsearch)
        self.storage = RAGStorage(
            chroma_collection_name=collection_name,
            es_index_name="query_es_dummy",  # placeholder
            chroma_path=chroma_path,
        )

    def add_query(self, query: str):
        """Add a new user query to the query collection."""
        doc_id = f"query_{abs(hash(query))}"
        response = self.storage.ollama.embeddings(model=self.storage.model_name, prompt=query)
        embedding = response["embedding"]

        self.storage.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[query],
            metadatas=[{"source": "user_query"}],
        )

    def find_similar(self, query: str, top_k: int = 1) -> Optional[str]:
        """Find the most semantically similar past query from ChromaDB."""
        response = self.storage.ollama.embeddings(model=self.storage.model_name, prompt=query)
        query_embedding = response["embedding"]

        results = self.storage.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        if not results["documents"] or not results["documents"][0]:
            return None
        return results["documents"][0][0]  # return most similar query


class RAGPromptBuilder:
    """
    Handles retrieval from Chroma and Elasticsearch,
    plus fallback to query history and LLM-based rewriting.
    """

    def __init__(
        self,
        chroma_collection_name: str,
        es_index_name: str,
        chroma_path: str = "../ingestion/vectorstore/chroma_db",

    ):
        cfg = load_config("./app/config/config.yaml")
        api_key = cfg["api"]["openai_key"]
        model_name = cfg["model"]["name_memory"]
        base_url = "https://api.openai.com/v1"

        self.llm = LLM(model=model_name, api_key=api_key, base_url=base_url)
        self.system_prompt = "You are a helpful assistant."
        self.storage = RAGStorage(
            chroma_collection_name=chroma_collection_name,
            es_index_name=es_index_name,
            chroma_path=chroma_path,
        )
        self.query_collection = QueryCollection(chroma_path)

    def retrieve_all(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve from both Chroma and Elasticsearch."""
        semantic_results = self.storage.semantic_search(query, top_k=top_k)
        es_results = self.storage.es_search(query, top_k=top_k)
        combined = semantic_results + es_results

        # deduplicate
        seen = set()
        unique = []
        for doc in combined:
            doc_id = doc.get("doc_id")
            if doc_id not in seen:
                unique.append(doc)
                seen.add(doc_id)
        return unique

    @staticmethod
    def build_prompt(retrieved_docs: List[Dict[str, Any]], user_question: str, max_context: int = 5) -> str:
        """Build the final prompt with retrieved context."""
        prompt_lines = ["Context (retrieved documents):"]
        for i, doc in enumerate(retrieved_docs[:max_context], start=1):
            doc_id = doc.get("doc_id", "")
            text = doc.get("full_text", "") or doc.get("text", "")
            url = doc.get("metadata", {}).get("url_file", "")
            prompt_lines.append(f"{i}. Doc ID: {doc_id}\n   {text}")
            if url:
                prompt_lines.append(f"   URL: {url}")
            prompt_lines.append("------")
        prompt_lines.append(f"\nUser question: {user_question}")
        prompt_lines.append("Answer based on the context above:")
        return "\n".join(prompt_lines)

    def retrieve_with_fallback(self, query: str, top_k: int = 3, max_context: int = 6) -> str:
        """
        Retrieval flow:
        1️⃣ Direct retrieval
        2️⃣ Fallback: similar query
        3️⃣ Fallback: LLM-rewritten query
        """
        # Step 1: Direct retrieval
        docs = self.retrieve_all(query, top_k=top_k)
        if docs:
            self.query_collection.add_query(query)
            return self.build_prompt(docs, query, max_context=max_context)

        # Step 2: Similar query
        similar_q = self.query_collection.find_similar(query)
        if similar_q:
            docs = self.retrieve_all(similar_q, top_k=top_k)
            if docs:
                print(f"[Fallback] Used similar query: {similar_q}")
                self.query_collection.add_query(query)
                return self.build_prompt(docs, query, max_context=max_context)

        # Step 3: Rewrite query with LLM
        rewrite_prompt = f"Rewrite this Persian query into a clearer, more general form:\n'{query}'"
        rewritten_q = self.llm.generate_response(
            system=self.system_prompt,
            prompt=rewrite_prompt,
            temperature=0.4,
            max_tokens=256
        )

        docs = self.retrieve_all(rewritten_q, top_k=top_k)
        if docs:
            print(f"[Fallback] Used rewritten query: {rewritten_q}")
            self.query_collection.add_query(query)
            self.query_collection.add_query(rewritten_q)
            return self.build_prompt(docs, query, max_context=max_context)

        return "هیچ سند مرتبطی یافت نشد."

    def retrieve_with_subqueries(self, query: str, top_k: int = 3, max_context: int = 6) -> str:
        """
        Step 1️⃣: Split a complex query into subqueries using LLM.
        Step 2️⃣: Retrieve docs for each subquery.
        Step 3️⃣: Merge + deduplicate results.
        """
        split_prompt = (
            f"Break this Persian query into smaller, independent sub-questions:\n"
            f"'{query}'\nReturn them as a JSON list (no extra text)."
        )

        try:
            subqueries_text = self.llm.generate_response(
                system=self.system_prompt,
                prompt=split_prompt,
                temperature=0.3,
                max_tokens=256,
            )
            import json
            subqueries = json.loads(subqueries_text)
            if not isinstance(subqueries, list):
                raise ValueError
        except Exception:
            subqueries = [query]  # fallback if LLM parsing fails

        all_docs = []
        for sub_q in subqueries:
            sub_docs = self.retrieve_all(sub_q, top_k=top_k)
            all_docs.extend(sub_docs)

        # deduplicate
        seen = set()
        unique_docs = []
        for d in all_docs:
            doc_id = d.get("doc_id")
            if doc_id not in seen:
                unique_docs.append(d)
                seen.add(doc_id)

        if not unique_docs:
            return self.retrieve_with_fallback(query, top_k=top_k, max_context=max_context)

        print(f"[Multi-step] Used {len(subqueries)} subqueries: {subqueries}")
        self.query_collection.add_query(query)
        return self.build_prompt(unique_docs, query, max_context=max_context)


if __name__ == "__main__":
    builder = RAGPromptBuilder(
        chroma_collection_name="rag_sam_d",
        es_index_name="rag_data",
    )

    question = "مراحل ثبت‌نام و انتخاب واحد برای دانشجوی مهمان در گلستان چیست؟"
    prompt = builder.retrieve_with_subqueries(question, top_k=3, max_context=6)
    print(prompt)
