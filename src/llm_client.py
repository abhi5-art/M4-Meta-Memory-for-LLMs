import json
import re
import httpx
from typing import List, Dict, Any, Optional
from src.config import AppConfig, setup_logger
from src.models import RawMemoryItem

logger = setup_logger("m4.llm_client")

class LLMClientException(Exception):
    """Raised when interaction with Ollama LLM / Embedding endpoints fails."""
    pass

class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.http_client = httpx.Client(timeout=120.0)

    def get_embedding(self, text: str) -> List[float]:
        """Calls native Ollama /api/embeddings endpoint."""
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/embeddings"
        payload = {
            "model": self.config.embedding_model,
            "prompt": text
        }
        try:
            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if "embedding" not in data:
                raise LLMClientException(f"Invalid embedding payload returned: {data}")
            return data["embedding"]
        except Exception as e:
            logger.error("Failed to generate embedding for input text: %s", str(e))
            raise LLMClientException(f"Embedding generation failure: {e}") from e

    def complete_chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Calls OpenAI-compatible /v1/chat/completions endpoint on Ollama."""
        url = f"{self.config.ollama_base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature
        }
        try:
            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Chat completion request failed: %s", str(e))
            raise LLMClientException(f"Chat completion failure: {e}") from e

    def extract_structured_json(self, prompt: str) -> List[RawMemoryItem]:
        """Requests extraction and validates JSON against schema with 1 retry."""
        messages = [
            {"role": "system", "content": "You are a specialized memory extraction engine. Respond strictly with a JSON array."},
            {"role": "user", "content": prompt}
        ]
        raw_text = self.complete_chat(messages, temperature=0.0)
        parsed = self._parse_json_array(raw_text)
        if parsed is not None:
            return parsed

        logger.warn("Initial JSON parse failed. Retrying with error feedback prompt...")
        retry_messages = [
            {"role": "system", "content": "You must correct invalid JSON. Output ONLY valid JSON array with objects matching keys 'Category' and 'Summary'."},
            {"role": "user", "content": f"The following output was invalid JSON:\n{raw_text}\nFix it and output ONLY the valid JSON array."}
        ]
        second_attempt = self.complete_chat(retry_messages, temperature=0.0)
        parsed_second = self._parse_json_array(second_attempt)
        if parsed_second is not None:
            return parsed_second

        raise LLMClientException(f"Failed to extract valid JSON memory clues after 1 retry. Output: {second_attempt}")

    def _parse_json_array(self, text: str) -> Optional[List[RawMemoryItem]]:
        clean_text = text.strip()
        # Strip markdown block ticks if present
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            array_match = re.search(r"\[.*\]", clean_text, re.DOTALL)
            if array_match:
                clean_text = array_match.group(0)

        try:
            data = json.loads(clean_text)
            if not isinstance(data, list):
                return None
            items = [RawMemoryItem.model_validate(item) for item in data]
            return items
        except Exception as ex:
            logger.debug("JSON array parse attempt rejected: %s", str(ex))
            return None
