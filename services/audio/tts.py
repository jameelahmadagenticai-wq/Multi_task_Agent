import hashlib
import tempfile
from pathlib import Path

try:
    from piper.voice import PiperVoice
except ImportError:
    PiperVoice = None


class TextToSpeech:

    def __init__(
        self,
        model_path: str = "models/en_US-lessac-medium.onnx"
    ):

        self.voice = None
        if PiperVoice is None or not Path(model_path).exists():
            return

        self.voice = PiperVoice.load(
            model_path
        )

    def synthesize(
        self,
        text: str
    ):
        if self.voice is None:
            return None

        output_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        with open(
            output_file.name,
            "wb"
        ) as wav_file:

            self.voice.synthesize(
                text,
                wav_file
            )

        return output_file.name
