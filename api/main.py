from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from api.middleware.logging import AuditMiddleware
from api.routers import agent, chat, upload


app = FastAPI(title="Air-Gapped Agentic AI API")

app.add_middleware(AuditMiddleware)
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(agent.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
