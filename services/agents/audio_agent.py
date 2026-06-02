from services.audio.stt import SpeechToText
from services.audio.tts import TextToSpeech
from services.audio.command_parser import CommandParser
from services.agents.orchestrator import Orchestrator


class AudioAgent:
    """
    Audio Agent Pipeline

    Audio
      ↓
    Speech To Text
      ↓
    Command Parser
      ↓
    Orchestrator
      ↓
    Text Response
      ↓
    Text To Speech
      ↓
    Audio Response
    """

    def __init__(self):

        self.stt = SpeechToText()

        self.tts = TextToSpeech()

        self.command_parser = CommandParser()

        self.orchestrator = Orchestrator()

    async def process_audio(
        self,
        audio_path: str,
        user_id: str = None,
        session_id: str = None
    ):
        """
        Complete voice pipeline.

        Returns:
        {
            transcript,
            intent,
            entities,
            answer,
            audio_response
        }
        """

        # ----------------------------------
        # STEP 1: Speech To Text
        # ----------------------------------

        transcription = self.stt.transcribe(
            audio_path
        )

        transcript = transcription["text"]

        # ----------------------------------
        # STEP 2: Parse Command
        # ----------------------------------

        command = await self.command_parser.parse_command(
            transcript
        )

        intent = command.get(
            "intent",
            "SEARCH"
        )

        entities = command.get(
            "entities",
            {}
        )

        confidence = command.get(
            "confidence",
            0.0
        )

        # ----------------------------------
        # STEP 3: Route To Agent System
        # ----------------------------------

        try:

            agent_response = await self.orchestrator.run(
                query=transcript,
                user_id=user_id,
                session_id=session_id
            )

        except Exception as error:

            agent_response = (
                f"Failed to process request: {str(error)}"
            )

        # ----------------------------------
        # STEP 4: Generate Voice Response
        # ----------------------------------

        audio_response = self.tts.synthesize(
            str(agent_response)
        )

        # ----------------------------------
        # Final Result
        # ----------------------------------

        return {
            "transcript": transcript,
            "intent": intent,
            "entities": entities,
            "confidence": confidence,
            "answer": agent_response,
            "audio_response": audio_response,
            "language": transcription.get(
                "language"
            ),
            "segments": transcription.get(
                "segments",
                []
            )
        }