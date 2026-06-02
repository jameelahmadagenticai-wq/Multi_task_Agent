import tempfile
import os

try:
    import ffmpeg
    from faster_whisper import WhisperModel
except ImportError:
    ffmpeg = None
    WhisperModel = None


class SpeechToText:

    def __init__(self):

        if WhisperModel is None:
            self.model = None
            return

        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

    def preprocess_audio(
        self,
        input_path: str
    ):
        if ffmpeg is None:
            return input_path

        output_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        (
            ffmpeg
            .input(input_path)
            .output(
                output_file.name,
                ac=1,
                ar=16000
            )
            .overwrite_output()
            .run(
                quiet=True
            )
        )

        return output_file.name

    def transcribe(
        self,
        audio_path: str
    ):
        if self.model is None:
            return {
                "text": "",
                "language": None,
                "segments": [],
                "error": "faster-whisper and ffmpeg-python are not installed.",
            }

        processed_audio = (
            self.preprocess_audio(
                audio_path
            )
        )

        segments, info = (
            self.model.transcribe(
                processed_audio
            )
        )

        transcript_segments = []

        full_text = []

        for segment in segments:

            transcript_segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
            )

            full_text.append(
                segment.text
            )

        if processed_audio != audio_path:
            os.remove(processed_audio)

        return {
            "text":
                " ".join(full_text),

            "language":
                info.language,

            "segments":
                transcript_segments
        }
