"""
utils.py


    from utils import load_config, load_system_prompt, trim_contexts, build_prompt
"""

import os
import logging
from pathlib import Path
import yaml
import textwrap
from typing import List, Dict, Any

# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
def load_config(config_path: str = "./config/config.yaml") -> dict:

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"فایل تنظیمات پیدا نشد: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)


    config["model"]["name_api"] = os.getenv("MODEL_NAME_API", config["model"].get("name_api"))
    config["api"]["openrouter_key"] = os.getenv("OPENROUTER_API_KEY", config["api"].get("openrouter_key"))
    config["elasticsearch"]["host"] = os.getenv("ES_HOST", config["elasticsearch"].get("host"))
    config["elasticsearch"]["index"] = os.getenv("ES_INDEX", config["elasticsearch"].get("index"))
    config["prompt"]["system_path"] = os.getenv("SYSTEM_PROMPT_PATH", config["prompt"].get("system_path"))
    config["limits"]["max_context_blocks"] = int(
        os.getenv("MAX_CONTEXT_BLOCKS", config["limits"].get("max_context_blocks", 5)))
    config["limits"]["max_context_chars"] = int(
        os.getenv("MAX_CONTEXT_CHARS", config["limits"].get("max_context_chars", 3000)))


    logger.info("تنظیمات از config.yaml بارگذاری شد.")
    return config


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def trim_contexts(aggregated_results: List[Dict[str, Any]], max_blocks: int, max_chars: int) -> List[Dict[str, Any]]:

    selected = aggregated_results[:max_blocks]
    total_chars = sum(len(r.get("text", "")) for r in selected)
    if total_chars <= max_chars:
        return selected

    ratio = max_chars / total_chars
    trimmed = []
    for r in selected:
        text = r.get("text", "").strip()
        max_len = max(200, int(len(text) * ratio))
        if len(text) > max_len:
            head = text[: max_len // 2]
            tail = text[-(max_len // 2):]
            new_text = head + "\n...\n" + tail
        else:
            new_text = text
        trimmed.append({"text": new_text, "metadata": r.get("metadata", {})})
    return trimmed


# ----------------------------------------------------------------------
def build_prompt(
    user_query: str,
    aggregated_results: List[Dict[str, Any]],
    system_prompt: str = None,
    max_context_blocks: int = 5,
    max_context_chars: int = 3000
) -> str:

    selected = trim_contexts(aggregated_results, max_context_blocks, max_context_chars)
    context_blocks = []
    for i, r in enumerate(selected, 1):
        text = r.get("text", "").strip()
        meta = r.get("metadata", {})
        context_blocks.append(f"Context {i}:\n{text}\nMetadata: {meta}\n")

    contexts = "\n".join(context_blocks) if context_blocks else "هیچ زمینه‌ای در دسترس نیست."

    system_text = f"{system_prompt}\n" if system_prompt else ""

    prompt = f"""
    {system_text}
    پرسش کاربر: {user_query}

    زمینه‌های مرتبط:
    {contexts}

    دستورالعمل‌ها:
    - فقط به فارسی رسمی پاسخ بده و فقط از زمینه‌های داده‌شده استفاده کن.
    - اگر اطلاعات کافی نیست، صراحتاً بگو "اطلاعات در دسترس نیست".
    - ابتدا با ۲–۴ جمله خلاصه شروع کن (بدون عنوان Markdown).
    - سپس جزئیات را با Markdown ساختاربندی کن (##، لیست، جدول).
    - اگر از زمینه نقل می‌کنی، منبع را ذکر کن (مثلاً: منبع: <نام فایل>).
    """
    return textwrap.dedent(prompt).strip()