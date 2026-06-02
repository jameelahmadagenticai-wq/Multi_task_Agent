import base64
import httpx
import asyncio

from config.settings import settings


class LLaVAAnalyzer:

    def __init__(self):

        self.url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    def encode_image(self, image_path):

        with open(image_path, "rb") as image_file:

            return base64.b64encode(
                image_file.read()
            ).decode("utf-8")

    async def analyze_image(
        self,
        image_path,
        prompt
    ):

        image_data = self.encode_image(image_path)

        payload = {
            "model": "llava:7b",
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "keep_alive": "10m"
        }

        timeout = httpx.Timeout(600.0)

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            response = await client.post(
                self.url,
                json=payload
            )

            response.raise_for_status()

            return response.json()["response"]


if __name__ == "__main__":

    async def main():

        analyzer = LLaVAAnalyzer()

        result = await analyzer.analyze_image(
            image_path="sample.png",
            prompt="Describe this image in detail."
        )

        print("\nLLaVA Response:\n")

        print(result)

    asyncio.run(main())
