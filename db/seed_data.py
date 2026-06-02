import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from services.rag.embedder import Embedder
from services.rag.chunker import TextChunker


sample_text = """
Artificial Intelligence allows machines to simulate human intelligence.

Machine learning is a subset of AI.

RAG systems combine retrieval and generation.

Vector databases store embeddings for semantic search.
"""


chunker = TextChunker()

chunks = chunker.chunk_text(sample_text)

embedder = Embedder()

vectors = embedder.embed(chunks)

client = QdrantClient(
    host="localhost",
    port=6333
)


points = []

for chunk, vector in zip(chunks, vectors):

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "text": chunk
            }
        )
    )


client.upsert(
    collection_name="text_chunks",
    points=points
)

print("Data inserted successfully")
