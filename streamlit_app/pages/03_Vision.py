import streamlit as st

import api_client
from components.file_uploader import image_uploader
from components.sidebar import render_sidebar


st.set_page_config(page_title="Vision", layout="wide")
render_sidebar()

st.title("Vision")

ocr_tab, qa_tab, chart_tab, screenshot_tab = st.tabs(
    ["OCR", "Image QA", "Chart Analysis", "Screenshot Debug"]
)

with ocr_tab:
    image = image_uploader("OCR image")
    if st.button("Extract text", disabled=not image):
        files = {"image": (image.name, image.getvalue(), image.type or "image/png")}
        try:
            response = api_client.post("/api/v1/vision/ocr", files=files)
            response.raise_for_status()
            payload = response.json()
            st.text_area("Extracted text", payload.get("text", ""), height=240)
            st.dataframe(payload.get("boxes", []), use_container_width=True)
        except Exception as exc:
            st.error(f"OCR failed: {exc}")

with qa_tab:
    image = image_uploader("QA image")
    question = st.text_input("Question", "What is shown in this image?")
    if st.button("Ask image", disabled=not image or not question):
        files = {"image": (image.name, image.getvalue(), image.type or "image/png")}
        try:
            response = api_client.post("/api/v1/vision/qa", files=files, data={"question": question})
            response.raise_for_status()
            st.write(response.json().get("answer"))
        except Exception as exc:
            st.error(f"Image QA failed: {exc}")

with chart_tab:
    image = image_uploader("Chart image")
    if st.button("Analyze chart", disabled=not image):
        files = {"image": (image.name, image.getvalue(), image.type or "image/png")}
        try:
            response = api_client.post("/api/v1/vision/analyze", files=files)
            response.raise_for_status()
            analysis = response.json().get("analysis", {})
            st.metric("Detected type", analysis.get("chart_type", "unknown"))
            st.dataframe(analysis.get("data_points", []), use_container_width=True)
            st.write(analysis.get("summary"))
        except Exception as exc:
            st.error(f"Chart analysis failed: {exc}")

with screenshot_tab:
    image = image_uploader("Screenshot")
    if st.button("Debug screenshot", disabled=not image):
        files = {"image": (image.name, image.getvalue(), image.type or "image/png")}
        try:
            response = api_client.post("/api/v1/vision/analyze", files=files)
            response.raise_for_status()
            st.write(response.json().get("analysis", {}).get("summary"))
        except Exception as exc:
            st.error(f"Screenshot analysis failed: {exc}")
