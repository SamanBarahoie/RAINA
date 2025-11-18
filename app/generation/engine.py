import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.agents.llm_local import LocalLLM
from app.agents.llm_memory import MemoryAgent
from app.utils.utils import load_config, build_prompt
from app.prompts.loader import load_prompt
from app.retrieval.ContextAggregator import RAGPromptBuilder
from app.agents.llm_api import LLM  # Optional API LLM
import requests  # to detect network errors

# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RAGAssistant:
    """RAG Assistant with proper source titles instead of Doc IDs."""

    def __init__(
        self,
        config_file: Path,
        chroma_collection_name: str = "rag_sam_d",
        es_index_name: str = "rag_data",
        chroma_path: str = "../vectorstore/chroma_db",
        use_local_llm: bool = False,
    ):
        # Load config
        try:
            self.cfg = load_config(config_file)
        except Exception as exc:
            logger.error("Error loading config: %s", exc)
            raise RuntimeError("Config loading failed") from exc

        self.use_local_llm = use_local_llm
        self.model_name_api = self.cfg["model"]["name_api"]
        self.model_name_local = self.cfg["model"]["name_local"]

        self.memory_agent = MemoryAgent(persist_dir="chroma_memory")

        self.rag_builder = RAGPromptBuilder(
            chroma_collection_name=chroma_collection_name,
            es_index_name=es_index_name,
            chroma_path=chroma_path,
        )

        self.system_prompt = load_prompt("system", "default")
        self.default_session_id = "default_session"
        self.default_user_id = None

    def _normalize_retrieved(self, raw: Any, max_context: int) -> List[Dict[str, Any]]:
        """
        Normalize retrieved docs into list of dicts with real source titles or URLs.
        """
        results: List[Dict[str, Any]] = []

        if isinstance(raw, str):
            results.append({
                "doc_id": "منبع نامشخص",
                "text": raw,
                "full_text": raw,
                "metadata": {},
            })
            return results[:max_context]

        if isinstance(raw, list):
            for item in raw[:max_context]:
                if isinstance(item, str):
                    results.append({
                        "doc_id": "منبع نامشخص",
                        "text": item,
                        "full_text": item,
                        "metadata": {},
                    })
                    continue

                if isinstance(item, dict):
                    text = item.get("text") or item.get("full_text") or item.get("summary") or ""
                    metadata = item.get("metadata", {})
                    # Use title or URL as doc_id for proper reference
                    doc_id = metadata.get("title") or metadata.get("source_name") or metadata.get("url_file") or text[:32]
                    results.append({
                        "doc_id": doc_id,
                        "text": text,
                        "full_text": item.get("full_text", text),
                        "metadata": metadata,
                    })
                    continue

                results.append({
                    "doc_id": "منبع نامشخص",
                    "text": str(item),
                    "full_text": str(item),
                    "metadata": {},
                })
            return results

        results.append({
            "doc_id": "منبع نامشخص",
            "text": str(raw),
            "full_text": str(raw),
            "metadata": {},
        })
        return results[:max_context]

    def retrieve_context(self, query: str, top_k: int = 3, max_context: int = 5) -> List[Dict[str, Any]]:
        try:
            raw = self.rag_builder.retrieve_with_subqueries(
                query=query, top_k=top_k, max_context=max_context
            )
        except Exception as exc:
            logger.exception("Retrieval failed: %s", exc)
            return []

        normalized = self._normalize_retrieved(raw, max_context=max_context)
        logger.debug("Normalized retrieved contexts: %s", normalized)
        return normalized

    @staticmethod
    def build_model_prompt(
            retrieved_docs: List[Dict],
            user_question: str,
            max_context: int = 5,
    ) -> str:
        """Build a readable Markdown prompt from retrieved docs with proper references."""

        def format_reference(doc: Dict[str, Any]) -> str:
            """Return a reference string using title and URL if available."""
            title = doc.get("metadata", {}).get("title") or doc.get("doc_id", "منبع نامشخص")
            url = doc.get("metadata", {}).get("url_file")
            if url:
                return f"[{title}]({url})"
            return title

        prompt_lines = ["Context (retrieved documents):"]
        for i, doc in enumerate(retrieved_docs[:max_context], start=1):
            full_text = doc.get("full_text", "") or doc.get("text", "")
            reference_str = format_reference(doc)

            prompt_lines.append(f"{i}. :منبع {reference_str}")
            prompt_lines.append(f"   :متن کامل {full_text}")
            prompt_lines.append("------")

        prompt_lines.append(f"\nUser question: {user_question}")
        prompt_lines.append("Answer based on the context above:")

        return "\n".join(prompt_lines)

    def generate_answer(self, user_question: str, session_id: Optional[str] = None, user_id: Optional[str] = None,
                        top_k: int = 3, max_context: int = 5) -> str:
        session_id = session_id or self.default_session_id
        user_id = user_id or self.default_user_id

        retrieved_docs = self.retrieve_context(user_question, top_k=top_k, max_context=max_context)
        logger.info("Retrieved %d contexts.", len(retrieved_docs))

        memory_context = ""
        combined_system_prompt = f"{self.system_prompt}\n\n{memory_context}"

        aggregated_results = retrieved_docs
        try:
            final_prompt = build_prompt(
                user_query=user_question,
                aggregated_results=aggregated_results,
                system_prompt=combined_system_prompt,
                max_context_blocks=self.cfg["limits"]["max_context_blocks"],
                max_context_chars=self.cfg["limits"]["max_context_chars"],
            )
        except Exception as exc:
            logger.warning("build_prompt failed (%s). Falling back to simple builder.", exc)
            final_prompt = self.build_model_prompt(aggregated_results, user_question, max_context=max_context)

        logger.debug("Final prompt length: %d", len(final_prompt))

        try:
            if self.use_local_llm:
                llm = LocalLLM(model_name=self.model_name_local)
                answer = llm.chat(final_prompt)
            else:
                api_llm = LLM(api_key=self.cfg["api"]["openai_key"], model=self.model_name_api)
                answer = api_llm.generate_response(prompt=final_prompt, system=combined_system_prompt)
        except requests.exceptions.RequestException as exc:
            logger.exception("Network error when calling LLM: %s", exc)
            raise
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            answer = "خطا در تولید پاسخ توسط مدل رخ داد."

        try:
            summary_text = f"پرسش: {user_question}\nپاسخ: {answer}"
            store_user_id = user_id or session_id
            self.memory_agent.store_memory(store_user_id, session_id, summary_text)

            json_path = Path("memory_exports") / f"{store_user_id}_{session_id}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_agent.export_user_memory_json(store_user_id, session_id, file_path=str(json_path))

            logger.info("Memory updated for user=%s, session=%s", store_user_id, session_id)
        except Exception as exc:
            logger.warning("Could not update memory: %s", exc)

        try:
            Path("last_prompt.txt").write_text(final_prompt, encoding="utf-8")
        except Exception:
            pass

        return answer

# Example usage (keep as needed)
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

    assistant = RAGAssistant(config_file=CONFIG_FILE, use_local_llm=False)
   # q = "مراحل ثبت نام ترم تابستان"
    q="با توجه به اطلاعیه معاونت آموزشی و تحصیلات تکمیلی دانشگاه سیستان و بلوچستان، اگر دانشجویی که درخواست مهمانی او در سامانه سجاد وزارت علوم تأیید شده است، پس از دریافت معرفی‌نامه از سامانه گلستان، در دانشگاه مقصد دروسی را انتخاب کند که در معرفی‌نامه ذکر نشده‌اند، چه پیامدهایی برای او به دنبال خواهد داشت؟ همچنین، مراحل دقیق ثبت درخواست معرفی‌نامه در سامانه گلستان و چگونگی ارتباط آن با سامانه سجاد را با جزئیات توضیح دهید."
    a = assistant.generate_answer(q)
    print(a)
