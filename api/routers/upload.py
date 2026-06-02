import asyncio
import hashlib
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sqlalchemy import func, select

from api.middleware.rbac import require_permission
from config.settings import settings
from db.models import Chunk, Document, IngestionLog
from db.postgres_client import PostgresManager


router = APIRouter(prefix="/api/v1", tags=["rag-vision"])
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "upload.bin").suffix
    path = Path(tempfile.gettempdir()) / f"aiagent_{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


@router.post("/rag/query")
async def rag_query(
    payload: QueryRequest,
    user: dict = Depends(require_permission("can_query")),
):
    from services.rag.rag_pipeline import RAGPipeline

    result = await RAGPipeline().ask(payload.query, top_k=payload.top_k)
    return {"query": payload.query, "top_k": payload.top_k, **result}


@router.post("/video/analyze")
async def analyze_video(
    file: UploadFile
):
    from services.agents.video_agent import VideoAgent

    path = _save_upload(file)
    try:
        agent = VideoAgent()
        result = await agent.analyze_video(str(path), video_id=uuid.uuid4().hex)
        return result
    finally:
        path.unlink(missing_ok=True)


@router.post("/rag/ingest")
async def rag_ingest(
    file: UploadFile = File(...),
    user: dict = Depends(require_permission("can_upload")),
):
    from services.rag.embedder import Embedder
    from services.rag.chunker import TextChunker

    content = await file.read()
    document_id = hashlib.sha256(content).hexdigest()
    filename = file.filename or "upload.txt"
    file_type = Path(filename).suffix.lower().lstrip(".") or "txt"

    existing = await _get_document_by_hash(document_id, user["user_id"])
    if existing:
        existing_document, existing_chunks = existing
        return {
            "document_id": existing_document.hash,
            "filename": existing_document.filename,
            "chunks": existing_chunks,
            "points_upserted": 0,
            "collection": "text_chunks",
            "status": "already_indexed",
        }

    text = await asyncio.to_thread(_extract_text, file.filename or "upload.txt", content)
    chunks = TextChunker().chunk_text(
        text,
        metadata={
            "doc_id": document_id,
            "source": filename,
            "user_id": user["user_id"],
            "type": file_type,
        },
    )

    if not chunks:
        await _write_ingestion_log(document_id, "empty", "No extractable text found.", user["user_id"])
        return {
            "document_id": document_id,
            "filename": filename,
            "chunks": 0,
            "points_upserted": 0,
            "status": "empty",
        }

    await _write_ingestion_log(document_id, "started", f"{filename} chunks={len(chunks)}", user["user_id"])
    _log_ingest(f"started filename={filename} chunks={len(chunks)}")
    embedder = Embedder()
    qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    _ensure_text_collection(qdrant)

    points_upserted = 0
    batch_size = 32
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        vectors = await asyncio.to_thread(embedder.embed, texts)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk['metadata']['chunk_index']}")),
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"],
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name="text_chunks",
            points=points,
            wait=True,
        )
        points_upserted += len(points)
        _log_ingest(f"batch_upserted filename={filename} upserted={points_upserted}/{len(chunks)}")

    await _persist_document(filename, document_id, file_type, chunks, user["user_id"])
    await _write_ingestion_log(document_id, "completed", f"points={points_upserted}", user["user_id"])
    _log_ingest(f"completed filename={filename} document_id={document_id} points={points_upserted}")
    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": len(chunks),
        "points_upserted": points_upserted,
        "collection": "text_chunks",
        "status": "indexed",
    }


@router.get("/documents")
async def list_documents(user: dict = Depends(require_permission("can_query"))):
    async with PostgresManager.get_session_factory()() as session:
        stmt = (
            select(
                Document.id,
                Document.filename,
                Document.hash,
                Document.type,
                Document.ingested_at,
                func.count(Chunk.id).label("chunks"),
            )
            .outerjoin(Chunk, Chunk.doc_id == Document.id)
            .where(Document.user_id == user["user_id"])
            .group_by(Document.id)
            .order_by(Document.ingested_at.desc())
        )
        rows = await session.execute(stmt)
        return {"documents": [dict(row._mapping) for row in rows]}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str, user: dict = Depends(require_permission("can_query"))):
    return {"job_id": job_id, "status": "completed", "progress": 100}


@router.post("/vision/ocr")
async def vision_ocr(
    image: UploadFile = File(...),
    user: dict = Depends(require_permission("can_query")),
):
    from services.vision.ocr import OCRProcessor

    path = _save_upload(image)
    try:
        result = OCRProcessor().extract_structured_text(str(path))
        return result
    finally:
        path.unlink(missing_ok=True)


@router.post("/vision/qa")
async def vision_qa(
    image: UploadFile = File(...),
    question: str = Form("Describe this image."),
    user: dict = Depends(require_permission("can_query")),
):
    from services.vision.image_qa import ImageQA

    path = _save_upload(image)
    try:
        answer = await ImageQA().ask([str(path)], question)
        return {"answer": answer, "question": question}
    finally:
        path.unlink(missing_ok=True)


@router.post("/vision/analyze")
async def vision_analyze(
    image: UploadFile = File(...),
    user: dict = Depends(require_permission("can_query")),
):
    from services.vision.chart_analyzer import ChartAnalyzer

    path = _save_upload(image)
    try:
        result = await ChartAnalyzer().analyze_chart(str(path))
        return {"analysis": result}
    finally:
        path.unlink(missing_ok=True)


@router.get("/vision/search")
async def vision_search(q: str, user: dict = Depends(require_permission("can_query"))):
    return {"query": q, "results": []}


def _ensure_text_collection(qdrant: QdrantClient):
    if qdrant.collection_exists("text_chunks"):
        return
    qdrant.create_collection(
        collection_name="text_chunks",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def _log_ingest(message: str):
    logger.info("rag_ingest_%s", message)
    print(f"[rag_ingest] {message}", flush=True)


async def _get_document_by_hash(document_hash: str, user_id: str) -> tuple[Document, int] | None:
    async with PostgresManager.get_session_factory()() as session:
        stmt = select(Document).where(Document.hash == document_hash, Document.user_id == user_id)
        result = await session.execute(stmt)
        document = result.scalar_one_or_none()
        if document is None:
            return None
        count_stmt = select(func.count(Chunk.id)).where(Chunk.doc_id == document.id)
        count_result = await session.execute(count_stmt)
        return document, int(count_result.scalar_one() or 0)


async def _persist_document(
    filename: str,
    document_hash: str,
    file_type: str,
    chunks: list[dict],
    user_id: str,
):
    async with PostgresManager.get_session_factory()() as session:
        document = Document(
            filename=filename,
            hash=document_hash,
            type=file_type,
            user_id=user_id,
        )
        document.chunks = [
            Chunk(
                page=chunk["metadata"].get("page") or chunk["metadata"].get("page_number"),
                text_preview=chunk["text"][:500],
            )
            for chunk in chunks
        ]
        session.add(document)
        await session.commit()


async def _write_ingestion_log(document_hash: str, status: str, message: str, user_id: str):
    try:
        async with PostgresManager.get_session_factory()() as session:
            session.add(
                IngestionLog(
                    document_hash=document_hash,
                    status=status,
                    message=message,
                    user_id=user_id,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.warning("ingestion_log_write_failed hash=%s error=%s", document_hash, exc)


def _extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(content)
    return content.decode("utf-8", errors="ignore")


def _extract_pdf_text(content: bytes) -> str:

    import pypdfium2 as pdfium

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:

        tmp.write(content)

        tmp_path = Path(tmp.name)

    try:

        pdf = pdfium.PdfDocument(str(tmp_path))

        pages = []

        try:

            for index, page in enumerate(pdf):

                text = page.get_textpage().get_text_range()

                if text.strip():

                    pages.append(
                        f"\n\n[Page {index + 1}]\n{text}"
                    )

        finally:

            pdf.close()

        return "\n".join(pages)

    finally:

        tmp_path.unlink(missing_ok=True)
