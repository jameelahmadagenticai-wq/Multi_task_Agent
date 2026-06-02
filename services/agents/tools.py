from functools import lru_cache


@lru_cache
def _ocr_processor():
    from services.vision.ocr import OCRProcessor

    return OCRProcessor()


@lru_cache
def _retriever():
    from services.rag.retriever import Retriever

    return Retriever()


def ocr_tool(image_path: str):
    return _ocr_processor().extract_text(image_path)


def retrieval_tool(query: str):
    results = _retriever().search(query)
    return [result.payload.get("text", "") for result in results]
