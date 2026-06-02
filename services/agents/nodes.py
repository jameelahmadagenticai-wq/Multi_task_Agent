from services.agents.state import AgentState
from services.agents.events import event_manager

from services.agents.tools import (
    ocr_tool,
    retrieval_tool
)


def router_node(state: AgentState):

    state["current_agent"] = "router_node"

    event_manager.emit(
        "agent_start",
        "router_node",
        "Router started"
    )

    print("\nRouter Node Running")

    query = state["user_query"].lower()

    image_keywords = [
        "image",
        "screenshot",
        "photo",
        "chart",
        "graph"
    ]

    if any(
        keyword in query
        for keyword in image_keywords
    ):

        state["task_type"] = "VISION"

    else:

        state["task_type"] = "RAG"

    state["history"].append({
        "agent": "router_node",
        "status": "completed"
    })

    event_manager.emit(
        "agent_done",
        "router_node",
        f"Task classified as {state['task_type']}"
    )

    return state






def retrieval_node(state: AgentState):

    state["current_agent"] = "retrieval_node"

    event_manager.emit(
        "agent_start",
        "retrieval_node",
        "Retrieval node started"
    )

    print("\nRetrieval Node Running")

    query = state["user_query"]

    event_manager.emit(
        "tool_call",
        "retrieval_node",
        "Calling retrieval tool"
    )

    docs = retrieval_tool(query)

    state["retrieved_docs"] = docs

    state["results"]["retrieval"] = docs

    state["history"].append({
        "agent": "retrieval_node",
        "status": "completed"
    })

    event_manager.emit(
        "agent_done",
        "retrieval_node",
        f"Retrieved {len(docs)} documents"
    )

    return state





def planner_node(state: AgentState):

    state["current_agent"] = "planner_node"

    print("\nPlanner Node Running")

    task_type = state["task_type"]

    if task_type == "VISION":

        execution_plan = [
            "vision",
            "reason",
            "critic",
            "memory"
        ]

    elif task_type == "RAG":

        execution_plan = [
            "retrieve",
            "reason",
            "critic",
            "memory"
        ]

    else:

        execution_plan = [
            "retrieve",
            "reason",
            "memory"
        ]

    state["execution_plan"] = execution_plan

    print("\nExecution Plan:\n")

    print(execution_plan)

    state["next_step"] = execution_plan[0]

    state["history"].append({
        "agent": "planner_node",
        "status": "completed"
    })

    return state


def reasoning_node(state: AgentState):

    state["current_agent"] = "reasoning_node"

    print("\nReasoning Node Running")

    docs = state["retrieved_docs"]

    iteration = state["iteration_count"]

    answer = f"""
    Iteration: {iteration}

    Based on retrieved docs:

    {docs}

    AI is a field where machines simulate intelligence.
    """

    state["final_answer"] = answer

    # enterprise structure
    state["results"]["reasoning"] = answer

    state["next_step"] = "critic"

    state["history"].append({
        "agent": "reasoning_node",
        "status": "completed"
    })

    return state


def critic_node(state: AgentState):

    state["current_agent"] = "critic_node"

    event_manager.emit(
        "agent_start",
        "critic_node",
        "Critic node started"
    )

    print("\nCritic Node Running")

    response = state.get("final_answer", "")

    if len(response) < 120:

        feedback = (
            "Response weak."
        )

    else:

        feedback = (
            "Response quality acceptable."
        )

    state["critic_feedback"] = feedback

    state["results"]["critic"] = feedback

    state["history"].append({
        "agent": "critic_node",
        "status": "completed"
    })

    event_manager.emit(
        "agent_done",
        "critic_node",
        feedback
    )

    return state





def vision_node(state: AgentState):

    state["current_agent"] = "vision_node"

    event_manager.emit(
        "agent_start",
        "vision_node",
        "Vision node started"
    )

    print("\nVision Node Running")

    event_manager.emit(
        "tool_call",
        "vision_node",
        "Calling OCR tool"
    )

    extracted_text = ocr_tool("sample.png")

    state["vision_result"] = extracted_text

    state["results"]["vision"] = extracted_text

    state["history"].append({
        "agent": "vision_node",
        "status": "completed"
    })

    event_manager.emit(
        "agent_done",
        "vision_node",
        "Vision analysis completed"
    )

    return state

memory_store = []


def memory_node(state: AgentState):

    state["current_agent"] = "memory_node"

    event_manager.emit(
        "agent_start",
        "memory_node",
        "Memory node started"
    )

    print("\nMemory Node Running")

    memory_entry = {
        "query": state["user_query"],
        "response": state.get("final_answer")
    }

    memory_store.append(memory_entry)

    state["results"]["memory"] = memory_entry

    state["status"] = "COMPLETED"

    state["history"].append({
        "agent": "memory_node",
        "status": "completed"
    })

    event_manager.emit(
        "agent_done",
        "memory_node",
        "Memory stored successfully"
    )

    return state