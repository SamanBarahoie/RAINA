"""
Airflow DAG: Ingestion Pipeline
PEP8 cleaned and fixed function-task mappings.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from app.ingestion.extract.extractor_manager import DownloadFolderProcessor
from app.ingestion.transform.transform import PersianRAGTransformer
from app.ingestion.load.loader import RAGStorage
from downloader import USBDownloaderMultipleFilters
from sanity_check import RagFailureAnalyzer


# -------------------------------
# Step Functions
# -------------------------------

def run_downloader() -> None:
    """Download documents using USBDownloaderMultipleFilters."""
    downloader = USBDownloaderMultipleFilters(category_filters=["فرآیندها"])
    downloader.run()


def run_extract() -> None:
    """Extract documents from downloaded folder."""
    processor = DownloadFolderProcessor("./data")
    processor.process_all()


def run_transform() -> None:
    """Transform extracted text into RAG dataset."""
    processor = PersianRAGTransformer(
        txt_folder="./data/txt",
        links_json="./data/downloaded_files.json",
        output_json="./data/rag_dataset_llm.json",
        chunk_size=400,
    )
    processor.process_documents()


def run_sanity_check() -> None:
    """Analyze failed chunks and save failure report."""
    analyzer = RagFailureAnalyzer(Path("./data/rag_dataset_llm.json"))
    failed = analyzer.analyze()

    print(failed)
    RagFailureAnalyzer.save_failure_report(
        failed, Path("./data/failed_docs.json")
    )


def run_retransform() -> None:
    """
    Re-run transformation only for documents that failed earlier.
    Reads './data/failed_docs.json'.
    """

    retry_file = Path("./data/failed_docs.json")

    if not retry_file.exists():
        print(f"{retry_file} not found. Nothing to reprocess.")
        return

    try:
        with retry_file.open("r", encoding="utf-8") as f:
            failed_docs = json.load(f)

        if not isinstance(failed_docs, list):
            raise ValueError("Retry file must contain a JSON list.")

    except Exception as exc:
        print(f"Failed to read {retry_file}: {exc}")
        return

    if not failed_docs:
        print("No failed documents to retry.")
        return

    print(f">>> Re-running transform for {len(failed_docs)} documents...")

    processor = PersianRAGTransformer(
        txt_folder="./data/txt",
        links_json="./data/downloaded_files.json",
        output_json="./data/rag_dataset_llm.json",
        chunk_size=400,
    )

    processor.process_specific_documents(failed_docs)

    print(">>> Re-run transform stage completed.")


def run_load() -> None:
    """Load cleaned documents into ChromaDB and Elasticsearch."""
    storage = RAGStorage(
        chroma_collection_name="rag_sam_d",
        es_index_name="rag_data",
    )

    data_path = Path("./data/rag_dataset_llm.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))

    storage.store(data)


# -------------------------------
# DAG Definition
# -------------------------------

default_args = {
    "execution_timeout": timedelta(hours=6),
    "retries": 0,
    "task_concurrency": 1,
}

with DAG(
    dag_id="ingestion_pipeline1",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
) as dag:

    download_task = PythonOperator(
        task_id="download_task",
        python_callable=run_downloader,
    )

    extract_task = PythonOperator(
        task_id="extract_task",
        python_callable=run_extract,
    )

    transform_task = PythonOperator(
        task_id="transform_task",
        python_callable=run_transform,
    )

    sanity_task = PythonOperator(
        task_id="sanity_task",
        python_callable=run_sanity_check,
    )

    retransform_task = PythonOperator(
        task_id="retransform_task",
        python_callable=run_retransform,
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=run_load,
    )

    (
        download_task >>
        extract_task >>
        transform_task >>
        sanity_task >>
        retransform_task >>
        load_task
    )
