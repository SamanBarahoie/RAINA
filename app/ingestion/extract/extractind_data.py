import logging
import tempfile
import subprocess
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pdf2image import convert_from_path
from app.utils.utils import load_config
from app.agents.llm_image import OpenAIOCR

# -------------------------
# Setup logger
# -------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Load config
# -------------------------
try:
    cfg = load_config("./app/config/config.yaml")
except Exception as e:
    logger.error("Error loading config: %s", e)
    raise RuntimeError("Config loading failed") from e

API_KEY = cfg["api"].get("openai_key")
MODEL_NAME = cfg["model"]["name_api"]

# -------------------------
# Utility functions
# -------------------------
def save_text(txt_path: Path, text: str) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Saved extracted text to %s", txt_path)


def init_ocr_client() -> OpenAIOCR:
    return OpenAIOCR(
        api_key=API_KEY,
        model=MODEL_NAME,
        system_prompt=(
            "You are an OCR assistant. Extract and clean Persian text "
            "from each image page with high accuracy and keep flowchart data readable."
        ),
        retries=5,
    )


def _make_safe_docx_copy(src: Path, temp_dir: Path) -> Path:
    """Copy src to a temp file with an ASCII-only, UUID-based name to avoid LibreOffice issues."""
    safe_name = f"doc_{uuid.uuid4().hex}.docx"
    safe_path = temp_dir / safe_name
    shutil.copy(src, safe_path)
    logger.info("Created safe copy for conversion: %s -> %s", src, safe_path)
    return safe_path

# -------------------------
# DOCX → PDF (Linux-safe)
# -------------------------
def convert_docx_to_pdf_linux(input_docx: Path, output_dir: Path, timeout: Optional[int] = 120) -> Path:
    """Convert DOCX to PDF using LibreOffice headless."""
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_pdf = output_dir / (input_docx.stem + ".pdf")

    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--norestore",
        "--nolockcheck",
        "--convert-to", "pdf",
        "--outdir", str(output_dir),
        str(input_docx),
    ]

    logger.info("Running LibreOffice command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        logger.error("LibreOffice timed out: %s", e)
        raise RuntimeError("LibreOffice conversion timed out") from e

    if not expected_pdf.exists() or expected_pdf.stat().st_size == 0:
        logger.error("❌ LibreOffice conversion failed or produced empty PDF")
        logger.error("STDOUT:\n%s", proc.stdout)
        logger.error("STDERR:\n%s", proc.stderr)
        raise RuntimeError("DOCX→PDF failed: PDF not created or empty")

    logger.info("✔ DOCX converted to PDF: %s (size=%d bytes)", expected_pdf, expected_pdf.stat().st_size)
    return expected_pdf

# -------------------------
# PDF OCR (page-by-page, low RAM)
# -------------------------
def process_pdf(pdf_path: Path, ocr_client=None, dpi: int = 150, max_pages: Optional[int] = None) -> str:
    logger.info("Processing PDF %s...", pdf_path)

    if ocr_client is None:
        ocr_client = init_ocr_client()

    text_data = ""
    temp_dir = Path(tempfile.gettempdir())
    temp_pdf = temp_dir / ("pdf_temp_" + uuid.uuid4().hex + ".pdf")
    try:
        shutil.copy(pdf_path, temp_pdf)
        logger.info("Copied PDF to temp file %s", temp_pdf)

        # Determine total pages
        try:
            total_pages = convert_from_path(temp_pdf, dpi=72, first_page=1, last_page=1)
            total_pages_count = len(total_pages)
        except Exception:
            total_pages_count = None  # fallback to None to process with pdf2image default
        if max_pages and total_pages_count:
            total_pages_count = min(total_pages_count, max_pages)

        # Page-by-page conversion
        page_idx = 1
        while True:
            if total_pages_count and page_idx > total_pages_count:
                break
            try:
                images = convert_from_path(temp_pdf, dpi=dpi, first_page=page_idx, last_page=page_idx)
            except Exception as e:
                logger.warning("Failed to convert PDF page %d: %s", page_idx, e)
                break
            if not images:
                break

            img = images[0]
            try:
                page_text = ocr_client.ocr(img, user_prompt="Extract and clean Persian text only:")
                if page_text:
                    text_data += f"\n\n--- Page {page_idx} ---\n{page_text.strip()}"
            except Exception as e:
                logger.warning("OCR failed on page %d: %s", page_idx, e)
            finally:
                img.close()  # free memory
            page_idx += 1

        save_text(pdf_path.with_suffix(".txt"), text_data)
        return text_data
    finally:
        if temp_pdf.exists():
            temp_pdf.unlink()
            logger.info("Deleted temporary PDF %s", temp_pdf)

# -------------------------
# Image OCR
# -------------------------
def process_image(image_path: Path, ocr_client=None) -> str:
    logger.info("Processing image: %s", image_path)
    img = Image.open(image_path)

    if ocr_client is None:
        ocr_client = init_ocr_client()

    text_data = ""
    try:
        text = ocr_client.ocr(img, user_prompt="Extract and clean Persian text only:")
        if text:
            text_data += text + "\n"
    except Exception as e:
        logger.warning("Failed OCR on image %s: %s", image_path.name, e)

    save_text(image_path.with_suffix(".txt"), text_data)
    return text_data

# -------------------------
# DOCX OCR
# -------------------------
def process_docx(docx_path: Path, ocr_client=None, dpi: int = 150, max_pages: Optional[int] = None) -> str:
    logger.info("Processing DOCX: %s", docx_path)

    if ocr_client is None:
        ocr_client = init_ocr_client()

    temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_docx = _make_safe_docx_copy(docx_path, temp_dir)

    try:
        temp_pdf = convert_docx_to_pdf_linux(safe_docx, temp_dir)
        text_data = process_pdf(temp_pdf, ocr_client=ocr_client, dpi=dpi, max_pages=max_pages)
        return text_data
    finally:
        if safe_docx.exists():
            safe_docx.unlink()
            logger.info("Deleted safe copy %s", safe_docx)
        if temp_pdf.exists():
            temp_pdf.unlink()
            logger.info("Deleted temporary PDF %s", temp_pdf)

# -------------------------
# Video placeholder
# -------------------------
def process_video(video_path: Path) -> str:
    text_data = video_path.name
    save_text(video_path.with_suffix(".txt"), text_data)
    return text_data

# -------------------------
# Main
# -------------------------
# if __name__ == "__main__":
#     # pdf_file = Path("فرآیند ثبت نام در ترم تابستان در سامانه جامع دانشگاهی گلستان.pdf")
#     #
#     ocr_client = init_ocr_client()
#     #
#     # process_pdf(pdf_file, ocr_client)
#
#     path_image = Path("../ﻗﺎﺑﻞ ﺗﻮﺟﻪ دانشجویان متقاضی مهمان به سایر دانشگاه ها.jpg")
#     process_image(path_image, ocr_client)
#    # process_docx(path_docx)