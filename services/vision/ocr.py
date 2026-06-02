import os

import cv2
import pytesseract


tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


class OCRProcessor:
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        threshold = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        return threshold

    def extract_text(self, image_path):
        processed = self.preprocess_image(image_path)
        return pytesseract.image_to_string(processed)

    def extract_structured_text(self, image_path):
        processed = self.preprocess_image(image_path)
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        boxes = []
        for index, text in enumerate(data.get("text", [])):
            if not text.strip():
                continue
            boxes.append(
                {
                    "text": text,
                    "confidence": float(data["conf"][index]),
                    "box": {
                        "x": int(data["left"][index]),
                        "y": int(data["top"][index]),
                        "width": int(data["width"][index]),
                        "height": int(data["height"][index]),
                    },
                }
            )
        return {"text": " ".join(box["text"] for box in boxes), "boxes": boxes}
