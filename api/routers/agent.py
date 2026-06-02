import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from api.middleware.rbac import require_permission
from services.agents.events import event_manager
from services.memory.short_term import ShortTermMemory

router = APIRouter(prefix="/api/v1", tags=["agents-memory"])
TASK_RESULTS: dict[str, dict] = {}


class AgentRunRequest(BaseModel):
    query: str
    context: dict = {}
    session_id: str = "default"


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/agent/run")
async def agent_run(
    payload: AgentRunRequest,
    user: dict = Depends(require_permission("can_query")),
):
    task_id = uuid.uuid4().hex
    state = {
        "task_id": task_id,
        "user_query": payload.query,
        "task_type": "",
        "execution_plan": [],
        "context": payload.context,
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
    from services.agents.workflow import graph

    result = await asyncio.to_thread(graph.invoke, state)
    TASK_RESULTS[task_id] = result
    memory = ShortTermMemory()
    await memory.add_turn(user["user_id"], payload.session_id, "user", payload.query)
    await memory.add_turn(
        user["user_id"],
        payload.session_id,
        "assistant",
        result.get("final_answer") or str(result.get("results", {})),
        agent_used=result.get("current_agent"),
    )
    return {
        "task_id": task_id,
        "status": result.get("status", "COMPLETED"),
        "final_answer": result.get("final_answer"),
        "history": result.get("history", []),
        "results": result.get("results", {}),
    }


@router.get("/agent/stream/{task_id}")
async def agent_stream(task_id: str, user: dict = Depends(require_permission("can_query"))):
    async def stream_events():
        result = TASK_RESULTS.get(task_id)
        events = event_manager.events if result else [{"event_type": "pending", "message": "Task not found"}]
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@router.get("/memory/recent")
async def recent_memory(
    session_id: str = "default",
    user: dict = Depends(require_permission("can_query")),
):
    return await ShortTermMemory().get_history(user["user_id"], session_id, last_n=10)


@router.delete("/memory/session")
async def clear_memory_session(
    session_id: str = "default",
    user: dict = Depends(require_permission("can_query")),
):
    await ShortTermMemory().clear_session(user["user_id"], session_id)
    return {"status": "cleared", "session_id": session_id}


@router.post("/memory/search")
async def search_memory(
    payload: MemorySearchRequest,
    user: dict = Depends(require_permission("can_query")),
):
    from services.memory.long_term import LongTermMemory

    memory = LongTermMemory()
    await memory.initialize()
    results = await memory.retrieve_similar(payload.query, user["user_id"], payload.top_k)
    return {"results": results}


@router.post("/audio/transcribe")
async def audio_transcribe(
    file: UploadFile = File(...),
    user: dict = Depends(require_permission("can_query")),
):
    from services.audio.stt import SpeechToText

    path = _save_upload(file)
    try:
        return SpeechToText().transcribe(str(path))
    finally:
        path.unlink(missing_ok=True)


@router.post("/audio/command")
async def audio_command(
    file: UploadFile = File(...),
    session_id: str = "default",
    user: dict = Depends(require_permission("can_query")),
):
    from services.agents.audio_agent import AudioAgent

    path = _save_upload(file)
    try:
        return await AudioAgent().process_audio(
            str(path),
            user_id=user["user_id"],
            session_id=session_id,
        )
    finally:
        path.unlink(missing_ok=True)


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.bin").suffix
    path = Path(tempfile.gettempdir()) / f"aiagent_{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path
