import json

from services.llm.ollama_client import OllamaClient


class CommandParser:

    SUPPORTED_INTENTS = [
        "ANALYZE_DOC",
        "SEARCH",
        "SUMMARIZE",
        "COMPARE",
        "OPEN_RECORD"
    ]

    def __init__(self):

        self.llm = OllamaClient()

    async def parse_command(
        self,
        transcript: str
    ):
        prompt = f"""
You are a voice command parser.

Supported intents:

{self.SUPPORTED_INTENTS}

Convert the user command into JSON.

Return ONLY valid JSON.

Schema:

{{
    "intent":"",
    "entities":{{}},
    "confidence":0.0,
    "raw_text":""
}}

User Command:

{transcript}
"""

        response = await self.llm.generate(
            prompt=prompt,
            model="phi3:mini"
        )

        try:
            parsed = json.loads(response)

            parsed["raw_text"] = transcript

            return parsed

        except Exception:

            return {
                "intent": "SEARCH",
                "entities": {
                    "query": transcript
                },
                "confidence": 0.50,
                "raw_text": transcript
            }