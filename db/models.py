from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    can_upload: Mapped[bool] = mapped_column(Boolean, default=False)
    can_query: Mapped[bool] = mapped_column(Boolean, default=True)
    can_admin: Mapped[bool] = mapped_column(Boolean, default=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    role_id: Mapped[int | None] = mapped_column(ForeignKey("roles.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user_id: Mapped[str | None] = mapped_column(String(255))

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    doc_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    page: Mapped[int | None] = mapped_column(Integer)
    text_preview: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[Document | None] = relationship(back_populates="chunks")


class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    document_hash: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user_id: Mapped[str | None] = mapped_column(String(255))


class Memory(Base):
    __tablename__ = "memory_store"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    importance_score: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentAudit(Base):
    __tablename__ = "agent_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str | None] = mapped_column(String(255))
    user_id: Mapped[str | None] = mapped_column(String(255))
    agent: Mapped[str | None] = mapped_column(Text)
    event_type: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(255))
    task_id: Mapped[str | None] = mapped_column(String(255))
    rating: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(255))
    endpoint: Mapped[str | None] = mapped_column(Text)
    response_code: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
