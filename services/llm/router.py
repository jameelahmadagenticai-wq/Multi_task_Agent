from pathlib import Path
from typing import AsyncIterator

import yaml

from services.llm.ollama_client import OllamaClient


class ModelRouter:
    def __init__(self, config_path: str = "services/llm/model_config.yaml"):
        with Path(config_path).open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.ollama = OllamaClient()

    def classify(self, prompt: str, task_type: str | None = None) -> str:
        if task_type:
            return task_type.upper()

        text = prompt.lower()
        if any(word in text for word in ("image", "photo", "screenshot", "chart", "ocr")):
            return "VISION"
        if any(word in text for word in ("summarize", "summary", "brief")):
            return "SUMMARIZE"
        if any(word in text for word in ("why", "reason", "compare", "analyze", "calculate")):
            return "REASON"
        if any(word in text for word in ("code", "python", "sql", "docker")):
            return "CODE"
        if any(word in text for word in ("audio", "voice", "transcribe")):
            return "AUDIO"
        return "CHAT"

    def route(self, task_type: str | None = None, prompt: str = "") -> dict:
        models = self.config["models"]
        route_key = {
            "CHAT": "fast",
            "FAST": "fast",
            "REASON": "reasoning",
            "VISION": "vision",
            "SUMMARIZE": "summarization",
            "CODE": "reasoning",
            "AUDIO": "fast",
        }.get(self.classify(prompt, task_type), "fast")
        return models[route_key]

    async def generate(
        self,
        prompt: str,
        task_type: str | None = None,
        stream: bool = False,
    ):
        model_config = self.route(task_type=task_type, prompt=prompt)
        options = {
            "temperature": model_config.get("temperature", 0.3),
            "num_predict": model_config.get("max_tokens", 512),
            "num_ctx": model_config.get("context_length", 4096),
            "timeout_seconds": model_config.get("timeout_seconds", 600),
        }
        result = await self.ollama.generate(
            model=model_config["name"],
            prompt=prompt,
            stream=stream,
            **options,
        )
        if stream:
            return result
        return result.get("response", "")

    async def stream(self, prompt: str, task_type: str | None = None) -> AsyncIterator[str]:
        stream = await self.generate(prompt=prompt, task_type=task_type, stream=True)
        async for chunk in stream:
            yield chunk
