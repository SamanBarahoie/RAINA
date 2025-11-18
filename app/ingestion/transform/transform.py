import json
import logging
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from app.agents.llm_api import LLM
from app.utils.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersianRAGTransformer:
    """
    Processes .txt files to produce RAG-compatible datasets.
    Skips already processed files, splits text into chunks, sends to LLM,
    and saves normalized JSON output.
    """

    def __init__(self, txt_folder: str, links_json: str, output_json: str, chunk_size: int = 400):
        cfg = load_config("./app/config/config.yaml")
        api_key = cfg["api"]["openai_key"]
        model_name = cfg["model"]["name_api"]
        base_url = "https://api.openai.com/v1"

        self.llm = LLM(model=model_name, api_key=api_key, base_url=base_url)
        self.txt_folder = Path(txt_folder)
        self.links_json = Path(links_json)
        self.output_json = Path(output_json)
        self.chunk_size = chunk_size

        # Load link metadata
        with open(self.links_json, "r", encoding="utf-8") as f:
            self.link_data = json.load(f)
        self.title_to_url = {i["title"]: i["url"] for i in self.link_data}

        # Load existing RAG dataset to avoid reprocessing
        self.existing_data = []
        self.existing_titles = set()
        if self.output_json.exists():
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    self.existing_data = json.load(f)
                    self.existing_titles = {chunk["metadata"]["title"] for chunk in self.existing_data}
            except json.JSONDecodeError:
                self.existing_data = []
                self.existing_titles = set()

        self.system_prompt = (
            "شما یک مدل زبانی هستید که برای استخراج داده‌های متنی فارسی برای سیستم‌های بازیابی و پاسخ‌گویی (RAG) طراحی شده‌اید. "
            "خروجی شما باید شامل چانک‌های متنی معنی‌دار، خلاصه، و متادیتا باشد."
        )

        self.user_prompt_template = (
            "عنوان سند: {doc_name}\n"
            "متن سند:\n{text}\n\n"
            "لطفاً متن بالا را به چند چانک منطقی تقسیم کنید (حداکثر {chunk_size} کلمه) و خروجی را به‌صورت JSON با ساختار زیر تولید کنید:\n"
            "[\n"
            "  {{\n"
            "    'chunk_text': '<بخش از متن>',\n"
            "    'metadata': {{\n"
            "       'title': '{doc_name}',\n"
            "       'page_range': [start_page, end_page] یا None,\n"
            "       'summary': '<خلاصه ۱ تا ۲ جمله‌ای>',\n"
            "       'topics': ['موضوع۱', 'موضوع۲']\n"
            "    }}\n"
            "  }}\n"
            "]\n\n"
            "پاسخ باید فقط شامل JSON معتبر باشد و متن اضافی ننویسید."
        )

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks of approx. self.chunk_size words."""
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def parse_llm_json(self, response: str) -> List[Dict[str, Any]]:
        """Safely parse JSON from LLM response, removing ```json ...``` blocks."""
        m = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        json_str = m.group(1) if m else response.strip()
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return [data]
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}; returning raw response wrapped as error.")
            return [{"error": "JSON parse failed", "raw": response, "exception": str(e)}]

    def _normalize_chunk(self, raw_chunk: Dict[str, Any], doc_name: str, url: str, chunk_id: int) -> Dict:
        """Normalize and enrich chunk output."""
        metadata = raw_chunk.get("metadata", {})
        return {
            "doc_id": doc_name,
            "chunk_id": chunk_id,
            "chunk_text": raw_chunk.get("chunk_text", ""),
            "metadata": {
                "title": metadata.get("title", doc_name),
                "url_file": url,
                "page_range": metadata.get("page_range", None),
                "summary": metadata.get("summary", ""),
                "topics": metadata.get("topics", []),
            },
        }

    def generate_report(self, new_chunks: List[Dict[str, Any]]):
        """Print summary report of processing."""
        total_new = len(new_chunks)
        total_duplicates = len(self.existing_titles)
        total_errors = sum(1 for chunk in new_chunks if "error" in chunk)

        logger.info("\n=== Processing Report ===")
        logger.info(f"New chunks processed: {total_new}")
        logger.info(f"Skipped duplicates: {total_duplicates}")
        logger.info(f"Failed chunks: {total_errors}")
        logger.info("========================\n")

    def process_documents(self):
        """Process all .txt files, skip already processed ones, and save JSON."""
        txt_files = list(self.txt_folder.glob("*.txt"))
        new_chunks = []

        for txt_file in tqdm(txt_files, desc="Processing documents", unit="doc"):
            doc_name = txt_file.stem
            if doc_name in self.existing_titles:
                logger.info(f"Skipping already processed file: {txt_file.name}")
                continue

            try:
                logger.info(f"Processing {txt_file.name}")
                text = txt_file.read_text(encoding="utf-8")

                # Match title → url
                title = next((t for t in self.title_to_url if t in doc_name or doc_name in t), doc_name)
                url = self.title_to_url.get(title, "")

                # Split text into chunks
                text_chunks = self._split_text(text)
                for t_chunk in text_chunks:
                    prompt = self.user_prompt_template.format(
                        doc_name=doc_name, text=t_chunk, chunk_size=self.chunk_size
                    )
                    response = self.llm.generate_response(
                        system=self.system_prompt,
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    raw_items = self.parse_llm_json(response)
                    for raw in raw_items:
                        normalized = self._normalize_chunk(raw, doc_name, url, chunk_id=len(self.existing_data) + len(new_chunks))
                        new_chunks.append(normalized)

            except Exception as e:
                logger.exception(f"Error processing {txt_file.name}: {e}")
                new_chunks.append({
                    "doc_id": doc_name,
                    "chunk_id": len(self.existing_data) + len(new_chunks),
                    "chunk_text": "",
                    "metadata": {"title": doc_name, "url_file": "", "page_range": None, "summary": None},
                    "error": str(e)
                })

        # Combine old and new chunks
        all_chunks = self.existing_data + new_chunks
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(new_chunks)} new chunks, total {len(all_chunks)} chunks to {self.output_json}")

        # Generate report
        self.generate_report(new_chunks)

    def process_specific_documents(self, doc_ids: List[str]):
        """Reprocess only specific documents and save JSON."""
        txt_files = [f for f in self.txt_folder.glob("*.txt") if f.stem in doc_ids]
        if not txt_files:
            logger.warning("No matching text files found for the given doc_ids.")
            return

        new_chunks = []

        for txt_file in tqdm(txt_files, desc="Reprocessing specific documents", unit="doc"):
            doc_name = txt_file.stem
            try:
                logger.info(f"Reprocessing {txt_file.name}")
                text = txt_file.read_text(encoding="utf-8")

                title = next((t for t in self.title_to_url if t in doc_name or doc_name in t), doc_name)
                url = self.title_to_url.get(title, "")

                text_chunks = self._split_text(text)
                for t_chunk in text_chunks:
                    prompt = self.user_prompt_template.format(
                        doc_name=doc_name, text=t_chunk, chunk_size=self.chunk_size
                    )
                    response = self.llm.generate_response(
                        system=self.system_prompt,
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    raw_items = self.parse_llm_json(response)
                    for raw in raw_items:
                        normalized = self._normalize_chunk(raw, doc_name, url, chunk_id=len(self.existing_data) + len(new_chunks))
                        new_chunks.append(normalized)

            except Exception as e:
                logger.exception(f"Error reprocessing {txt_file.name}: {e}")
                new_chunks.append({
                    "doc_id": doc_name,
                    "chunk_id": len(self.existing_data) + len(new_chunks),
                    "chunk_text": "",
                    "metadata": {"title": doc_name, "url_file": "", "page_range": None, "summary": None},
                    "error": str(e)
                })

        all_chunks = self.existing_data + new_chunks
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Reprocessed {len(new_chunks)} chunks, total {len(all_chunks)} chunks to {self.output_json}")

        # Generate report
        self.generate_report(new_chunks)


if __name__ == "__main__":
    processor = PersianRAGTransformer(
        txt_folder="../downloads/txt",
        links_json="../downloads/downloaded_files.json",
        output_json="../downloads/rag_dataset_llm.json",
        chunk_size=400
    )
    processor.process_documents()
