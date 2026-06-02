from services.llm.ollama_client import OllamaClient


class VideoSummarizer:

    def __init__(self):
        self.llm = OllamaClient()

    async def summarize_video(
        self,
        frame_descriptions: list
    ):
        combined_text = "\n".join(
            frame_descriptions
        )

        prompt = f"""
Summarize this video.

Scene descriptions:

{combined_text}

Provide:

1. Main events
2. Important actions
3. Final summary
"""

        summary = await self.llm.generate(
            prompt=prompt,
            model="mistral:7b"
        )

        return summary