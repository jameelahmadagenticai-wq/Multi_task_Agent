from services.llm.ollama_client import OllamaClient


class SceneDetector:

    SCENE_CATEGORIES = [
        "person",
        "machine",
        "vehicle",
        "text-on-screen",
        "anomaly"
    ]

    def __init__(self):
        self.llm = OllamaClient()

    async def classify_scene(
        self,
        frame_path: str
    ):
        prompt = f"""
Analyze this image.

Choose ONE category:

{self.SCENE_CATEGORIES}

Return JSON:

{{
    "category":"",
    "description":""
}}
"""

        response = await self.llm.generate_vision(
            image_path=frame_path,
            prompt=prompt
        )

        return response