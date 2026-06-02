from services.memory.short_term import ShortTermMemory
from services.memory.long_term import LongTermMemory


class ContextManager:

    def __init__(self):

        self.short_memory = ShortTermMemory()

        self.long_memory = LongTermMemory()

    async def initialize(self):

        await self.long_memory.initialize()

    async def build_context(
        self,
        user_id: str,
        session_id: str,
        current_query: str
    ):

        short_history = await self.short_memory.get_history(
            user_id=user_id,
            session_id=session_id,
            last_n=10
        )

        long_memories = await self.long_memory.retrieve_similar(
            query_text=current_query,
            user_id=user_id,
            top_k=5
        )

        context = {
            "recent_conversation": short_history,
            "relevant_memories": long_memories
        }

        return context