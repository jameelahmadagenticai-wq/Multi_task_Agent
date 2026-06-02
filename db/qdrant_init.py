from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import settings

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


collections = [
    ("text_chunks", 384),
    ("image_embeddings", 512),
    ("video_frames", 512),
    ("memory_store", 384),
]


for name, size in collections:

    existing = client.collection_exists(name)

    if existing:
        print(f"{name} already exists")
        continue

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=size,
            distance=Distance.COSINE
        )
    )

    print(f"{name} created successfully")
