# extractor_manager.py
import logging
from pathlib import Path
from typing import Optional
from .extractind_data  import *
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DownloadFolderProcessor:
    """
    Process all files in the downloads directory (PDF, DOCX, JPG, MP4)
    and save all extracted text into a 'txt' subfolder.
    """

    def __init__(self, download_dir: str):
        self.download_dir = Path(download_dir)
        self.txt_dir = self.download_dir / "txt"
        self.ocr_client = init_ocr_client()

        # Create output directory
        self.txt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized processor for %s", self.download_dir)

    def process_all(self):
        """Process all files inside known subdirectories."""
        for subdir in ["pdf", "docx", "jpg", "mp4"]:
            folder = self.download_dir / subdir
            if not folder.exists():
                logger.warning("Folder %s does not exist, skipping.", folder)
                continue

            logger.info("Processing folder: %s", folder)
            for file_path in folder.glob("*.*"):
                try:
                    text = self.process_file(file_path)
                    if text:
                        out_path = self.txt_dir / f"{file_path.stem}.txt"
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        logger.info("✅ Saved %s", out_path.name)
                except Exception as e:
                    logger.error("❌ Failed to process %s: %s", file_path.name, e)

    def process_file(self, file_path: Path) -> Optional[str]:
        """Detect file type and route to proper processing method."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return process_pdf(file_path, self.ocr_client)
        elif suffix == ".jpg" or suffix == ".jpeg" or suffix == ".png":
            return process_image(file_path, self.ocr_client)
        elif suffix == ".docx":
            return process_docx(file_path, self.ocr_client)
        elif suffix == ".mp4":
            return process_video(file_path)
        else:
            logger.warning("Unsupported file type: %s", suffix)
            return None


if __name__ == "__main__":
    processor = DownloadFolderProcessor("../downloads")
    processor.process_all()
