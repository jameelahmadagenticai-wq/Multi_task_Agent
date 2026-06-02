import asyncio
import uuid

from services.agents.workflow import graph


class Orchestrator:
    async def run(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        context: dict | None = None,
    ):
        task_id = uuid.uuid4().hex
        state = {
            "task_id": task_id,
            "user_query": query,
            "task_type": "",
            "execution_plan": [],
            "context": {
                "user_id": user_id,
                "session_id": session_id,
                **(context or {}),
            },
            "results": {},
            "history": [],
            "current_agent": None,
            "final_answer": None,
            "status": "RUNNING",
            "error": None,
            "iteration_count": 0,
            "retrieved_docs": [],
            "vision_result": None,
            "critic_feedback": None,
            "next_step": None,
        }
        result = await asyncio.to_thread(graph.invoke, state)
        return result.get("final_answer") or result.get("results", {})
