from services.memory.long_term import LongTermMemory
from services.llm.router import ModelRouter


class MemoryAgent:

    def __init__(self):

        self.long_memory = LongTermMemory()

        self.router = ModelRouter()

    async def initialize(self):

        await self.long_memory.initialize()

    async def evaluate_importance(
        self,
        text: str
    ):

        prompt = f"""
        You are a memory classifier.

        Score how important this information is
        for future conversations.

        Return ONLY a float between 0 and 1.

        Text:
        {text}
        """

        response = await self.router.generate(
            prompt=prompt,
            task_type="REASON"
        )

        try:
            score = float(response.strip())
        except:
            score = 0.3

        return score

    async def process_memory(
        self,
        user_id: str,
        conversation_text: str
    ):

        importance_score = await self.evaluate_importance(
            conversation_text
        )

        if importance_score > 0.7:

            await self.long_memory.store_memory(
                content=conversation_text,
                user_id=user_id,
                importance_score=importance_score
            )

            return {
                "stored": True,
                "importance_score": importance_score
            }

        return {
            "stored": False,
            "importance_score": importance_score
        }