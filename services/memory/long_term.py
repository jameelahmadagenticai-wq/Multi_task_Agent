from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from db.postgres_client import PostgresManager
from db.models import Memory


class LongTermMemory:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.session_factory = PostgresManager.get_session_factory()

    async def initialize(self):
        self.session_factory = PostgresManager.get_session_factory()

    async def generate_embedding(self, text: str):

        embedding = self.model.encode(text)

        return embedding.tolist()

    async def store_memory(
        self,
        content: str,
        user_id: str,
        importance_score: float = 0.5
    ):

        embedding = await self.generate_embedding(content)

        async with self.session_factory() as session:
            session.add(
                Memory(
                    user_id=user_id,
                    content=content,
                    embedding=embedding,
                    importance_score=importance_score,
                )
            )
            await session.commit()

    async def retrieve_similar(
        self,
        query_text: str,
        user_id: str,
        top_k: int = 5
    ):

        query_embedding = await self.generate_embedding(
            query_text
        )

        sql = text("""
        SELECT
            content,
            importance_score,
            created_at,
            embedding <=> CAST(:embedding AS vector) AS distance
        FROM memory_store
        WHERE user_id = :user_id
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """)

        async with self.session_factory() as session:
            rows = await session.execute(
                sql,
                {"embedding": query_embedding, "user_id": user_id, "top_k": top_k},
            )

        return [dict(row._mapping) for row in rows]
