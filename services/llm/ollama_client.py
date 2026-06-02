import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import httpx

from config.settings import settings


logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **options,
    ):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": "30m",
            "options": options,
        }
        if stream:
            return self.stream_generate(model=model, prompt=prompt, **options)
        return await self._post_json("/api/generate", payload)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **options,
    ):
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "30m",
            "options": options,
        }
        if stream:
            return self.stream_chat(model=model, messages=messages, **options)
        return await self._post_json("/api/chat", payload)

    async def stream_generate(
        self,
        model: str,
        prompt: str,
        **options,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": options,
        }
        async for chunk in self._stream_json("/api/generate", payload):
            if chunk.get("response"):
                yield chunk["response"]

    async def stream_chat(
        self,
        model: str,
        messages: list[dict],
        **options,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": options,
        }
        async for chunk in self._stream_json("/api/chat", payload):
            content = chunk.get("message", {}).get("content")
            if content:
                yield content

    async def _post_json(self, path: str, payload: dict):
        timeout = httpx.Timeout(payload.get("options", {}).get("timeout_seconds", 600.0))
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f"{self.base_url}{path}", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    self._log_usage(payload["model"], data)
                    return data
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

    async def _stream_json(self, path: str, payload: dict) -> AsyncIterator[dict]:
        timeout = httpx.Timeout(payload.get("options", {}).get("timeout_seconds", 600.0))
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", f"{self.base_url}{path}", json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line:
                                data = json.loads(line)
                                self._log_usage(payload["model"], data)
                                yield data
                return
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

    def _log_usage(self, model: str, data: dict):
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        if prompt_tokens or completion_tokens:
            logger.info(
                "ollama_usage model=%s prompt_tokens=%s completion_tokens=%s",
                model,
                prompt_tokens,
                completion_tokens,
            )
