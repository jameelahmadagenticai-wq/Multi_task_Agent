from services.video.frame_extractor import FrameExtractor
from services.video.scene_detector import SceneDetector
from services.video.video_summarizer import VideoSummarizer


class VideoAgent:

    def __init__(self):

        self.extractor = FrameExtractor()

        self.scene_detector = SceneDetector()

        self.summarizer = VideoSummarizer()

    async def analyze_video(
        self,
        video_path: str,
        video_id: str
    ):

        frames = self.extractor.process_video(
            video_path,
            video_id
        )

        for frame in frames:

            scene_info = await self.scene_detector.classify_scene(
                frame["frame_path"]
            )

            frame["scene_info"] = scene_info
            if isinstance(scene_info, dict):
                frame["description"] = scene_info.get("description", "")
            else:
                frame["description"] = str(scene_info)

        summary = await self.summarizer.summarize_video(
            frame_descriptions=[
                f"Frame {frame.get('frame_number', index)}: {frame.get('description', '')}"
                for index, frame in enumerate(frames)
            ]
        )

        return {
            "frames": frames,
            "summary": summary
        }
    
    
