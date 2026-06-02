from services.llm.router import ModelRouter
from services.rag.reranker import Reranker


class RAGPipeline:
    def __init__(self):
        self.reranker = Reranker()
        self.router = ModelRouter()

    async def ask(self, query: str, top_k: int = 5):
        results = self.reranker.rerank(query, top_k=top_k)
        context_blocks = []
        sources = []

        for score, result in results:
            payload = result.payload or {}
            text = payload.get("text", "")
            context_blocks.append(text)
            sources.append(
                {
                    "score": float(score),
                    "source": payload.get("source"),
                    "page": payload.get("page") or payload.get("page_number"),
                    "text_preview": text[:240],
                }
            )

        prompt = f"""
Answer the question using only the provided context. If the answer is not in
the context, say you do not know from the indexed documents.

Context:
{chr(10).join(context_blocks)}

Question:
{query}
"""
        answer = await self.router.generate(task_type="CHAT", prompt=prompt)
        return {"answer": answer, "sources": sources}
