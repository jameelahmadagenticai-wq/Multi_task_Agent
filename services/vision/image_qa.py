import base64
from pathlib import Path

from services.llm.ollama_client import OllamaClient


class ImageQA:
    def __init__(self):
        self.ollama = OllamaClient()

    def _encode_image(self, image_path: str):
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    async def ask(self, image_paths: list[str], question: str):
        images = [self._encode_image(path) for path in image_paths[:4]]
        payload = {
            "model": "llava:7b",
            "prompt": question,
            "images": images,
            "stream": False,
            "keep_alive": "10m",
        }
        result = await self.ollama._post_json("/api/generate", payload)
        return result.get("response", "")
