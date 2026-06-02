from services.rag.rag_pipeline import RAGPipeline


class RAGAgent:
    async def run(self, query: str, top_k: int = 5) -> dict:
        return await RAGPipeline().ask(query, top_k=top_k)


async def rag_agent(state: dict) -> dict:
    result = await RAGAgent().run(state["user_query"])
    state.setdefault("results", {})["rag"] = result
    state["final_answer"] = result.get("answer")
    state.setdefault("history", []).append({"agent": "rag_agent", "status": "completed"})
    return state
