# local_llm.py
from typing import Optional, Any, Dict, List
import logging
import ollama

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LocalLLM:
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = "You are RAINA; follow provided rules. Be formal, factual, and Persian.",
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _build_messages(self, prompt: str, extra_system: Optional[str] = None) -> List[Dict[str, str]]:
        sys_text = self.system_prompt or ""
        if extra_system:
            sys_text = sys_text.rstrip() + "\n" + extra_system.lstrip()
        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _extract_answer(response: Any) -> str:
        """
        Compatible with both dict and Ollama ChatResponse objects.
        """
        if response is None:
            return ""

        # Handle Ollama's new response object format
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content

        # Handle older dict-style responses
        if isinstance(response, dict):
            msg = response.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            if "output" in response:
                out = response["output"]
                if isinstance(out, list):
                    return "\n".join(map(str, out))
                return str(out)
            return str(response)

        # fallback: stringify the object
        return str(response)

    def chat(
        self,
        prompt: str,
        extra_system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        messages = self._build_messages(prompt, extra_system)
        call_kwargs: Dict[str, Any] = {"model": self.model_name, "messages": messages}
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        call_kwargs.update(kwargs)

        try:
            logger.debug("Calling ollama.chat with args: %s", call_kwargs)
            response = ollama.chat(**call_kwargs)
            answer = self._extract_answer(response)
            return answer.strip()
        except Exception as e:
            logger.exception("Error calling ollama.chat: %s", e)
            raise

    def chat_raw(self, **call_kwargs) -> Any:
        try:
            return ollama.chat(**call_kwargs)
        except Exception as e:
            logger.exception("Error in chat_raw: %s", e)
            raise
