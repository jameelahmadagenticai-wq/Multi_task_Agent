from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        base_metadata = metadata or {}
        chunks = self.splitter.split_text(text or "")
        return [
            {
                "text": chunk,
                "metadata": {
                    **base_metadata,
                    "chunk_index": index,
                },
            }
            for index, chunk in enumerate(chunks)
        ]

    def chunk_file(self, path: str, metadata: dict[str, Any] | None = None):
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return self.chunk_text(
            text,
            {
                "source": file_path.name,
                "type": file_path.suffix.lower().lstrip(".") or "txt",
                **(metadata or {}),
            },
        )
