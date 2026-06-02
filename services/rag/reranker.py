from sentence_transformers import CrossEncoder

from services.rag.retriever import Retriever


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.retriever = Retriever()

    def rerank(
        self,
        query: str,
        top_k=5
    ):

        results = self.retriever.search(
            query=query,
            limit=10
        )

        pairs = []

        for result in results:

            text = result.payload["text"]

            pairs.append((query, text))

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True
        )

        return ranked[:top_k]


if __name__ == "__main__":

    reranker = Reranker()

    results = reranker.rerank(
        "What is Artificial Intelligence?"
    )

    for score, result in results:

        print("\nRerank Score:", score)

        print("Text:", result.payload["text"])