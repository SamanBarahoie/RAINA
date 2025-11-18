import io
import os
import base64
import time
import logging
from pathlib import Path
from typing import Any, Dict, Union
from PIL import Image
from openai import OpenAI

logger = logging.getLogger("OpenAIOCR")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_TIMEOUT = 180
MAX_RETRIES = 3
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

class OpenAIOCR:
    """
    OCR client using OpenAI multimodal models (e.g. gpt-4o-mini).
    Handles:
      - Base64 encoding
      - Proper payload for OpenAI
      - Retry on transient network or HTTP errors
      - Fallback for max_completion_tokens
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are an OCR assistant. Extract and clean Persian text from images.",
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = MAX_RETRIES,
    ):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image: Union[str, Path, bytes, Image.Image]) -> str:
        """Convert input image to base64 data URI."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            with open(path, "rb") as f:
                b = f.read()
            ext = path.suffix.lstrip(".") or "png"
        elif isinstance(image, bytes):
            b = image
            ext = "png"
        elif isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b = buf.getvalue()
            ext = "png"
        else:
            raise ValueError("Unsupported image type for OCR (expected Path, bytes, or PIL.Image).")

        encoded = base64.b64encode(b).decode("utf-8")
        return f"data:image/{ext};base64,{encoded}"

    def _extract_content(self, response) -> str:
        """Validate and extract OCR text content from the chat response."""
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
        else:
            raise RuntimeError("Invalid response: missing content in choices[0].message")
        return content

    def ocr(
        self,
        image: Union[str, Path, bytes, Image.Image],
        user_prompt: str = "Extract Persian text and return it cleanly:",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Send image to OpenAI and extract Persian text with retries and fallback."""
        image_data_uri = self._encode_image(image)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ]

        attempt = 0
        backoff = 3
        last_exception = None

        while attempt < self.retries:
            attempt += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = self._extract_content(response)
                if not text:
                    logger.warning("Empty OCR result (attempt %d)", attempt)
                return text

            except Exception as e:
                status = getattr(e, "status_code", "Unknown")
                resp_text = str(e)
                logger.error("Error on attempt %d: %s", attempt, resp_text)

                if status in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                last_exception = e
                break

        raise RuntimeError(f"OCR failed after {self.retries} retries. Last error: {last_exception}")