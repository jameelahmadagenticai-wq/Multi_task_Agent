import os
import uuid
import cv2
import torch

from PIL import Image
from qdrant_client.http.models import Distance, VectorParams

try:
    from scenedetect import ContentDetector, detect
except ImportError:
    ContentDetector = None
    detect = None

from transformers import (
    CLIPModel,
    CLIPProcessor
)

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


class FrameExtractor:
    """
    Video Processing Pipeline

    Step 1:
        - Scene Detection
        - Key Frame Extraction

    Step 2:
        - CLIP Embeddings
        - Qdrant Indexing
    """

    MAX_FRAMES = 200

    def __init__(
        self,
        output_dir: str = "data/video_frames",
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
    ):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.clip_model = None
        self.clip_processor = None

        self.qdrant = QdrantClient(
            host=qdrant_host,
            port=qdrant_port
        )
        self._ensure_collection()

    # ==================================================
    # Scene Detection
    # ==================================================

    def detect_scenes(
        self,
        video_path: str
    ):
        """
        Detect scene boundaries.

        Returns:
            Scene list from PySceneDetect
        """

        if detect is None or ContentDetector is None:
            return []

        return detect(video_path, ContentDetector(threshold=27.0))

    # ==================================================
    # Frame Extraction
    # ==================================================

    def extract_key_frames(
        self,
        video_path: str
    ):
        """
        Extract one frame per scene.

        Returns:
            List[dict]
        """

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            raise Exception(
                f"Cannot open video: {video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        ) or 30

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0

        frame_metadata = []
        scenes = self.detect_scenes(video_path) or self._fallback_scenes(duration)

        for scene_id, scene in enumerate(
            scenes
        ):
            if len(frame_metadata) >= self.MAX_FRAMES:
                break

            start_time, end_time = self._scene_seconds(scene)

            midpoint_time = (
                start_time + end_time
            ) / 2

            frame_number = int(
                midpoint_time * fps
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = cap.read()

            if not success:
                continue

            filename = (
                f"scene_{scene_id}_"
                f"{int(midpoint_time)}.jpg"
            )

            frame_path = os.path.join(
                self.output_dir,
                filename
            )

            cv2.imwrite(
                frame_path,
                frame
            )

            frame_metadata.append(
                {
                    "frame_path": frame_path,
                    "frame_number": frame_number,
                    "timestamp_sec": round(
                        midpoint_time,
                        2
                    ),
                    "scene_id": scene_id,
                }
            )

        cap.release()

        return frame_metadata

    def _fallback_scenes(self, duration: float):
        if duration <= 0:
            return [(0, 1)]
        step = 30
        return [
            (start, min(start + step, duration))
            for start in range(0, int(duration), step)
        ][: self.MAX_FRAMES]

    def _scene_seconds(self, scene):
        start, end = scene
        if hasattr(start, "get_seconds"):
            return start.get_seconds(), end.get_seconds()
        return float(start), float(end)

    # ==================================================
    # CLIP Embedding
    # ==================================================

    def generate_embedding(
        self,
        image_path: str
    ):
        """
        Generate CLIP embedding.

        Returns:
            List[float]
        """
        self._load_clip()

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():
            image_features = (
                self.clip_model.get_image_features(
                    **inputs
                )
            )

        embedding = (
            image_features
            .squeeze()
            .cpu()
            .numpy()
            .tolist()
        )

        return embedding

    def _load_clip(self):
        if self.clip_model is not None and self.clip_processor is not None:
            return

        print("Loading CLIP model...")

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        print("CLIP model loaded.")

    def _ensure_collection(self):
        if self.qdrant.collection_exists("video_frames"):
            return

        self.qdrant.create_collection(
            collection_name="video_frames",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    # ==================================================
    # Qdrant Indexing
    # ==================================================

    def index_frame(
        self,
        frame_info: dict,
        embedding: list,
        point_id: str,
    ):
        """
        Store frame vector in Qdrant.
        """

        self.qdrant.upsert(
            collection_name="video_frames",
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "video_id":
                            frame_info["video_id"],

                        "timestamp_sec":
                            frame_info["timestamp_sec"],

                        "scene_id":
                            frame_info["scene_id"],

                        "frame_path":
                            frame_info["frame_path"],
                    }
                )
            ]
        )

    # ==================================================
    # Complete Processing Pipeline
    # ==================================================

    def process_video(
        self,
        video_path: str,
        video_id: str
    ):
        """
        Complete pipeline:

        Video
            ->
        Scene Detection
            ->
        Frame Extraction
            ->
        CLIP Embedding
            ->
        Qdrant
        """

        frames = self.extract_key_frames(
            video_path
        )

        indexed_frames = []

        for frame in frames:

            frame["video_id"] = video_id

            embedding = (
                self.generate_embedding(
                    frame["frame_path"]
                )
            )

            point_id = str(
                uuid.uuid4()
            )

            self.index_frame(
                frame,
                embedding,
                point_id
            )

            indexed_frames.append(
                frame
            )

        return indexed_frames


# ======================================================
# Local Testing
# ======================================================

if __name__ == "__main__":

    extractor = FrameExtractor()

    results = extractor.process_video(
        video_path="sample.mp4",
        video_id="video_001"
    )

    print("\nIndexed Frames:\n")

    for item in results:
        print(item)
