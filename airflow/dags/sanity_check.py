from pathlib import Path
from typing import Dict, Set, List
import json


REQUIRED_TOPLEVEL_KEYS = {"doc_id", "chunk_id", "chunk_text", "metadata"}
REQUIRED_METADATA_KEYS = {"title", "page_range", "summary", "topics"}


class RagFailureAnalyzer:
    """
    Analyze RAG JSON entries and extract failed doc_ids with reasons.
    """

    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.data = self._load_json()

    # --------------------------------------------------------------
    def _load_json(self) -> List[dict]:
        """Load input RAG JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.json_path}")

        with self.json_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # --------------------------------------------------------------
    def analyze(self) -> Dict[str, Set[str]]:
        """
        Scan the JSON and return: doc_id -> set(reasons)
        """
        failures: Dict[str, Set[str]] = {}

        for i, chunk in enumerate(self.data):
            doc_id = str(chunk.get("doc_id") or f"__missing_docid_chunk_{i}")

            if doc_id not in failures:
                failures[doc_id] = set()

            # 1) Error field
            if chunk.get("error"):
                failures[doc_id].add("error")

            # 2) Missing top-level keys
            missing_top = REQUIRED_TOPLEVEL_KEYS - set(chunk.keys())
            if missing_top:
                failures[doc_id].add("missing_toplevel")

            # 3) Empty chunk_text
            chunk_text = chunk.get("chunk_text", "")
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                failures[doc_id].add("empty_chunk_text")

            # 4) Metadata
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                failures[doc_id].add("missing_metadata")
            else:
                # missing metadata keys
                missing_meta = REQUIRED_METADATA_KEYS - set(metadata.keys())
                if missing_meta:
                    failures[doc_id].add("missing_metadata")

                # invalid topics
                topics = metadata.get("topics")
                if topics is not None and not isinstance(topics, list):
                    failures[doc_id].add("invalid_topics")

        # remove doc_ids with no issues
        cleaned = {did: reasons for did, reasons in failures.items() if reasons}
        return cleaned

    # --------------------------------------------------------------
    @staticmethod
    def save_failure_report(failed_map: Dict[str, Set[str]], output_path: Path) -> None:
        """
        Save doc_id → reasons as JSON.
        """
        out_list = [{"doc_id": doc_id, "reasons": sorted(list(reasons))}
                    for doc_id, reasons in sorted(failed_map.items())]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(out_list, fh, ensure_ascii=False, indent=2)

    # --------------------------------------------------------------
    @staticmethod
    def build_txt_retry_list(failed_map: Dict[str, Set[str]], txt_folder: Path) -> List[str]:
        """
        Match doc_ids to txt files and return retry paths.
        """
        txt_files = {p.stem: p for p in txt_folder.glob("*.txt")}
        retry_paths = []

        for doc_id in sorted(failed_map.keys()):
            if doc_id in txt_files:
                retry_paths.append(str(txt_files[doc_id].resolve()))
            else:
                retry_paths.append(doc_id)

        return retry_paths
