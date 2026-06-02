from services.vision.llava_analyzer import LLaVAAnalyzer


class ChartAnalyzer:

    def __init__(self):

        self.llava = LLaVAAnalyzer()

    async def analyze_chart(
        self,
        image_path
    ):

        prompt = """
        Analyze this chart or screenshot carefully.

        Identify:
        - chart type
        - trends
        - anomalies
        - labels
        - errors
        - important insights

        Explain findings clearly.
        """

        response = await self.llava.analyze_image(
            image_path=image_path,
            prompt=prompt
        )

        chart_type = "unknown"
        lowered = response.lower()
        for candidate in ("bar", "line", "pie", "table"):
            if candidate in lowered:
                chart_type = candidate
                break

        return {
            "chart_type": chart_type,
            "data_points": [],
            "summary": response,
        }


if __name__ == "__main__":

    import asyncio

    async def main():

        analyzer = ChartAnalyzer()

        result = await analyzer.analyze_chart(
            "sample1.png"
        )

        print("\nChart Analysis:\n")

        print(result)

    asyncio.run(main())
