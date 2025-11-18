import os


# -------------------------
# Configuration
# -------------------------
PROMPT_DIR = os.path.dirname(__file__)


def load_prompt(category: str, name: str) -> str:
    """
    Load a text prompt by category and name.

    Example:
        load_prompt("system", "default")
    will load:
        app/prompts/system/default.txt
    """
    path = os.path.join(PROMPT_DIR, category, f"{name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
