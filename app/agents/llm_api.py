from typing import List, Dict, Optional, Any
import requests
import time
import logging

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
logger = logging.getLogger(__name__)


class LLM:
    """
    Unified chat client compatible with OpenAI and OpenRouter.
    Supports retries, exponential backoff, automatic fallback for
    'max_completion_tokens', and optional temperature handling.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        # Auto-detect base URL if not provided
        self.base_url = (base_url or OPENAI_API_BASE).rstrip("/")
        self.is_openrouter = "openrouter.ai" in self.base_url

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: int,
        use_completion_tokens: bool = False,
    ) -> Dict[str, Any]:
        """
        Build request payload.
        OpenRouter may not support 'temperature' in all models.
        """
        payload = {
            "model": self.model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature

        if use_completion_tokens:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        return payload

    def _extract_content(self, data: Dict[str, Any]) -> str:
        """
        Robust extraction of assistant text from OpenAI/OpenRouter response.
        """
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"Missing 'choices' in response: {data}")
        first = choices[0]
        # Newer responses: choices[0]["message"]["content"]
        msg = first.get("message") if isinstance(first, dict) else None
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"].strip()
        # Fallbacks: choices[0]["text"] or other shapes
        text = first.get("text") or first.get("message", {}).get("content", "")
        return (text or "").strip()

    def generate_response(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1024,
        retries: int = 3,
        backoff: float = 2.0,
        timeout: int = 60,
    ) -> str:
        """
        Send a chat/completions request with robust error handling.
        - Auto-detects OpenRouter vs OpenAI.
        - Retries on transient errors with exponential backoff.
        - Falls back to 'max_completion_tokens' if needed.
        """
        if not self.api_key:
            raise RuntimeError("API key is not set for LLM client.")

        if messages:
            payload_messages = messages
        else:
            if system is None or prompt is None:
                raise ValueError(
                    "If `messages` is not provided, both `system` and `prompt` must be supplied."
                )
            payload_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        current_backoff = backoff
        tried_alternate_param = False

        while attempt < retries:
            attempt += 1
            payload = self._build_payload(
                messages=payload_messages,
                temperature=temperature if temperature is not None else None,
                max_tokens=max_tokens,
                use_completion_tokens=tried_alternate_param,
            )

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                return self._extract_content(data)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                resp_text = e.response.text if e.response else str(e)

                if status == 401:
                    logger.error("Unauthorized (401). Check your API key and base_url.")
                    raise RuntimeError(f"Unauthorized (401): {resp_text}") from e

                if status == 400 and ("max_tokens" in resp_text or "unsupported_parameter" in resp_text):
                    if not tried_alternate_param:
                        logger.warning("Model rejected 'max_tokens'. Retrying with 'max_completion_tokens'.")
                        tried_alternate_param = True
                        continue
                    raise RuntimeError(f"OpenAI/OpenRouter HTTP 400: {resp_text}") from e

                if status in (429, 502, 503, 504):
                    if attempt < retries:
                        logger.warning(
                            "HTTP %s. Retrying %d/%d after %.1fs...",
                            status, attempt, retries, current_backoff
                        )
                        time.sleep(current_backoff)
                        current_backoff *= 2
                        continue
                    raise RuntimeError(f"HTTP error after retries: {status} - {resp_text}") from e

                raise RuntimeError(f"HTTP error: {status} - {resp_text}") from e

            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    logger.warning("Network error: %s. Retrying %d/%d after %.1fs...", e, attempt, retries, current_backoff)
                    time.sleep(current_backoff)
                    current_backoff *= 2
                    continue
                raise RuntimeError(f"Network error when calling LLM: {e}") from e

            except Exception as e:
                raise RuntimeError(f"Unexpected error when contacting LLM: {e}") from e

        raise RuntimeError("Failed to obtain response after multiple retries.")
