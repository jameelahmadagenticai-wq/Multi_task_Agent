from services.vision.image_qa import ImageQA
from services.vision.ocr import OCRProcessor


class VisionAgent:
    async def run(self, image_path: str, question: str | None = None) -> dict:
        if question:
            answer = await ImageQA().ask([image_path], question)
            return {"answer": answer, "mode": "qa"}
        return OCRProcessor().extract_structured_text(image_path)


async def vision_agent(state: dict) -> dict:
    image_path = state.get("context", {}).get("image_path")
    if not image_path:
        state.setdefault("results", {})["vision"] = {"error": "No image_path provided."}
        return state
    result = await VisionAgent().run(image_path, state.get("user_query"))
    state.setdefault("results", {})["vision"] = result
    state.setdefault("history", []).append({"agent": "vision_agent", "status": "completed"})
    return state
