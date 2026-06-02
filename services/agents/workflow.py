from langgraph.graph import StateGraph, END

from services.agents.state import AgentState

from services.agents.nodes import (
    router_node,
    planner_node,
    retrieval_node,
    reasoning_node,
    vision_node,
    memory_node,
    critic_node
)

from services.agents.analyst_node import analyst_node


workflow = StateGraph(AgentState)


# nodes
workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("vision", vision_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("critic", critic_node)
workflow.add_node("memory", memory_node)


# entry point
workflow.set_entry_point("router")


# execution flow
workflow.add_edge("router", "planner")

workflow.add_conditional_edges(
    "planner",
    lambda state: state["task_type"],
    {
        "RAG": "retrieve",
        "VISION": "vision"
    }
)

# merge branches
workflow.add_edge("retrieve", "analyst")
workflow.add_edge("vision", "analyst")

# validation flow
workflow.add_edge("analyst", "critic")
workflow.add_edge("critic", "memory")

# end workflow
workflow.add_edge("memory", END)


# compile graph
graph = workflow.compile()


if __name__ == "__main__":

    initial_state = {

        "task_id": "task_001",

        "user_query": "What is AI?",

        "task_type": "",

        "execution_plan": [],

        "context": {},

        "results": {},

        "history": [],

        "current_agent": None,

        "final_answer": None,

        "status": "RUNNING",

        "error": None,

        "iteration_count": 0,

        # compatibility fields
        "retrieved_docs": [],

        "vision_result": None,

        "critic_feedback": None,

        "next_step": None
    }

    result = graph.invoke(initial_state)

    print("\nFinal State:\n")

    print(result)