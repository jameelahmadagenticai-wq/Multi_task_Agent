from services.vision.ocr import OCRProcessor
from services.vision.llava_analyzer import LLaVAAnalyzer


class MultimodalPipeline:

    def __init__(self):

        self.ocr = OCRProcessor()

        self.llava = LLaVAAnalyzer()

    async def analyze(
        self,
        image_path
    ):

        extracted_text = self.ocr.extract_text(
            image_path
        )

        vision_analysis = await self.llava.analyze_image(
            image_path=image_path,
            prompt="""
            Analyze this image carefully.

            Understand:
            - layout
            - charts
            - UI
            - errors
            - relationships
            - important visual insights
            """
        )

        combined_result = f"""
        OCR Extracted Text:
        {extracted_text}

        Vision Analysis:
        {vision_analysis}
        """

        return combined_result


if __name__ == "__main__":

    import asyncio

    async def main():

        pipeline = MultimodalPipeline()

        result = await pipeline.analyze(
            "sample.png"
        )

        print("\nMultimodal Analysis:\n")

        print(result)

    asyncio.run(main())