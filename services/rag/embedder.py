from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed(self, texts):

        embeddings = self.model.encode(texts)

        return embeddings


if __name__ == "__main__":

    embedder = Embedder()

    vectors = embedder.embed([
        "What is AI?",
        "Artificial Intelligence explained"
    ])

    print(vectors.shape)