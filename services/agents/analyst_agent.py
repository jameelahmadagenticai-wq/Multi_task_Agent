from services.llm.router import ModelRouter


class AnalystAgent:
    async def synthesize(self, query: str, results: dict) -> str:
        prompt = f"""
Synthesize a concise answer for the user request using these agent results.

Request:
{query}

Agent results:
{results}
"""
        return await ModelRouter().generate(prompt=prompt, task_type="REASON")


async def analyst_agent(state: dict) -> dict:
    answer = await AnalystAgent().synthesize(
        state.get("user_query", ""),
        state.get("results", {}),
    )
    state["final_answer"] = answer
    state.setdefault("history", []).append({"agent": "analyst_agent", "status": "completed"})
    return state
