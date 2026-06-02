import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.middleware.auth import AuthManager
from api.middleware.rbac import require_permission
from services.llm.router import ModelRouter
from services.memory.short_term import ShortTermMemory


router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = logging.getLogger(__name__)


class TokenRequest(BaseModel):
    username: str = "local_user"
    role: str = "user"


class GenerateRequest(BaseModel):
    prompt: str
    task_type: str | None = None
    stream: bool = False
    session_id: str = "default"


@router.post("/auth/token")
async def create_token(payload: TokenRequest):
    token = AuthManager.create_access_token(
        {"sub": payload.username, "username": payload.username, "role": payload.role}
    )
    return {"access_token": token, "token_type": "bearer", "role": payload.role}


@router.post("/llm/generate")
async def generate(
    payload: GenerateRequest,
    user: dict = Depends(require_permission("can_query")),
):
    router_service = ModelRouter()
    memory = ShortTermMemory()
    await _add_turn_safely(memory, user["user_id"], payload.session_id, "user", payload.prompt)

    if payload.stream:
        async def stream_response():
            answer_parts = []
            async for chunk in router_service.stream(payload.prompt, payload.task_type):
                answer_parts.append(chunk)
                yield chunk
            await _add_turn_safely(
                memory,
                user["user_id"],
                payload.session_id,
                "assistant",
                "".join(answer_parts),
                agent_used="llm_router",
            )

        return StreamingResponse(stream_response(), media_type="text/plain")

    try:
        answer = await router_service.generate(
            prompt=payload.prompt,
            task_type=payload.task_type,
        )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable. Start Ollama or set OLLAMA_BASE_URL.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {exc.response.text}",
        ) from exc

    await _add_turn_safely(
        memory,
        user["user_id"],
        payload.session_id,
        "assistant",
        answer,
        agent_used="llm_router",
    )
    return {"answer": answer, "task_type": router_service.classify(payload.prompt, payload.task_type)}


async def _add_turn_safely(memory: ShortTermMemory, *args, **kwargs):
    try:
        await memory.add_turn(*args, **kwargs)
    except Exception as exc:
        logger.warning("short_term_memory_write_failed error=%s", exc)
