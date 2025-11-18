"""Extract failed doc_ids from a RAG JSON and save them for reprocessing.

Usage:
    python extract_failed_docs.py --input ../downloads/rag_dataset_llm.json \
        --output ../downloads/failed_docs_to_retry.json
"""
from pathlib import Path
import json
import argparse
from typing import Dict, Set, List


REQUIRED_TOPLEVEL_KEYS = {"doc_id", "chunk_id", "chunk_text", "metadata"}
REQUIRED_METADATA_KEYS = {"title", "page_range", "summary", "topics"}


def analyze_rag_json(input_path: Path) -> Dict[str, Set[str]]:
    """
    Scan the RAG JSON and return a mapping doc_id -> set(reasons).

    Reasons can include:
      - "error"             : chunk contains an 'error' key with a message
      - "empty_chunk_text"  : chunk_text is empty or whitespace
      - "missing_toplevel"  : missing any top-level required key
      - "missing_metadata"  : missing metadata key(s)
      - "invalid_topics"    : metadata['topics'] exists but is not a list
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    failures: Dict[str, Set[str]] = {}

    for i, chunk in enumerate(data):
        # Safely get doc_id (use placeholder if missing)
        doc_id = str(chunk.get("doc_id") or f"__missing_docid_chunk_{i}")

        # Ensure set exists
        if doc_id not in failures:
            failures[doc_id] = set()

        # 1) explicit error field
        err = chunk.get("error")
        if err:
            failures[doc_id].add("error")

        # 2) top-level keys
        missing_top = REQUIRED_TOPLEVEL_KEYS - set(chunk.keys())
        if missing_top:
            failures[doc_id].add("missing_toplevel")

        # 3) chunk_text empty
        chunk_text = chunk.get("chunk_text", "")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            failures[doc_id].add("empty_chunk_text")

        # 4) metadata checks
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            failures[doc_id].add("missing_metadata")
        else:
            missing_meta = REQUIRED_METADATA_KEYS - set(metadata.keys())
            if missing_meta:
                failures[doc_id].add("missing_metadata")

            topics = metadata.get("topics")
            if topics is not None and not isinstance(topics, list):
                failures[doc_id].add("invalid_topics")

        # If for a doc_id there are no reasons collected, remove the key later.
        # (we keep temporarily; will filter after loop)

    # remove doc_ids with empty reason sets (no problems)
    cleaned: Dict[str, Set[str]] = {
        did: reasons for did, reasons in failures.items() if reasons
    }
    return cleaned


def save_failed_docs(failed_map: Dict[str, Set[str]], output_path: Path) -> None:
    """Save the failed doc_ids map as a JSON list of objects with reasons."""
    out_list: List[Dict[str, List[str]]] = []
    for doc_id, reasons in sorted(failed_map.items()):
        out_list.append({"doc_id": doc_id, "reasons": sorted(list(reasons))})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(out_list, fh, ensure_ascii=False, indent=2)


def build_txt_retry_list(failed_map: Dict[str, Set[str]], txt_folder: Path) -> List[str]:
    """
    (Optional) Build list of txt file paths for retrying,
    matching doc_id -> txt file stem. If not found, include doc_id itself.
    """
    txt_files = {p.stem: p for p in txt_folder.glob("*.txt")}
    retry_paths: List[str] = []
    for doc_id in sorted(failed_map.keys()):
        if doc_id in txt_files:
            retry_paths.append(str(txt_files[doc_id].resolve()))
        else:
            # if no matching txt file, still include doc_id for manual handling
            retry_paths.append(doc_id)
    return retry_paths


def main():
    parser = argparse.ArgumentParser(description="Extract failed doc_ids for reprocessing.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input RAG JSON (e.g. ../downloads/rag_dataset_llm.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../downloads/failed_docs_to_retry.json",
        help="Path to output JSON with failed doc_ids",
    )
    parser.add_argument(
        "--txt-folder",
        "-t",
        type=str,
        default="../downloads/txt",
        help="(optional) folder of original txt files to produce retry paths",
    )
    parser.add_argument(
        "--save-txt-list",
        action="store_true",
        help="If set, also write <output>.txt_list.json containing paths or doc_ids to retry",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_map = analyze_rag_json(input_path)

    if not failed_map:
        print("No failed chunks detected. Nothing to save.")
        return

    save_failed_docs(failed_map, output_path)
    print(f"Saved {len(failed_map)} failed doc_ids to {output_path}")

    if args.save_txt_list:
        txt_folder = Path(args.txt_folder)
        retry_list = build_txt_retry_list(failed_map, txt_folder)
        txt_list_path = output_path.with_name(output_path.stem + ".txt_list.json")
        txt_list_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_list_path.open("w", encoding="utf-8") as fh:
            json.dump(retry_list, fh, ensure_ascii=False, indent=2)
        print(f"Saved txt retry list to {txt_list_path}")


if __name__ == "__main__":
    main()

# python json_test.py -i ../downloads/rag_dataset_llm.json -o ../downloads/failed_docs_to_retry.json  --save-txt-list -t ../downloads/
