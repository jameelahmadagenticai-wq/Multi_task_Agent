from services.agents.state import AgentState

from services.agents.events import event_manager


def analyst_node(state: AgentState):

    state["current_agent"] = "analyst_node"

    event_manager.emit(
        "agent_start",
        "analyst_node",
        "Analyst node started"
    )

    print("\nAnalyst Node Running")

    retrieval_result = state["results"].get(
        "retrieval",
        []
    )

    vision_result = state["results"].get(
        "vision",
        ""
    )

    critic_feedback = state.get(
        "critic_feedback",
        ""
    )

    final_answer = f"""
    Retrieval Result:
    {retrieval_result}

    Vision Result:
    {vision_result}

    Critic Feedback:
    {critic_feedback}
    """

    state["final_answer"] = final_answer

    state["results"]["analyst"] = final_answer

    state["history"].append({
        "agent": "analyst_node",
        "status": "completed"
    })

    event_manager.emit(
        "agent_done",
        "analyst_node",
        "Final synthesis completed"
    )

    return state