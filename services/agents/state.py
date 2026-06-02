from typing import (
    TypedDict,
    List,
    Dict,
    Any,
    Optional
)


class AgentState(TypedDict):

    task_id: str

    user_query: str

    task_type: str

    execution_plan: List[str]

    context: Dict[str, Any]

    results: Dict[str, Any]

    history: List[Dict[str, Any]]

    current_agent: Optional[str]

    final_answer: Optional[str]

    status: str

    error: Optional[str]

    iteration_count: int

    # compatibility fields
    retrieved_docs: List[str]

    vision_result: Optional[str]

    critic_feedback: Optional[str]

    next_step: Optional[str]