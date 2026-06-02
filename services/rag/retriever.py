from qdrant_client import QdrantClient

from config.settings import settings
from services.rag.embedder import Embedder


class Retriever:

    def __init__(self):

        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        self.embedder = Embedder()

    def search(
        self,
        query: str,
        collection_name="text_chunks",
        limit=5
    ):

        query_vector = self.embedder.embed([query])[0]

        results = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=limit
        ).points

        return results


if __name__ == "__main__":

    retriever = Retriever()

    results = retriever.search(
        "Artificial Intelligence"
    )

    for result in results:

        print("\nScore:", result.score)

        print("Payload:", result.payload)
