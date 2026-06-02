import pandas as pd
import streamlit as st

import api_client
from components.sidebar import render_sidebar


st.set_page_config(page_title="Video", layout="wide")
render_sidebar()

st.title("Video")

video = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
if st.button("Analyze video", disabled=not video):
    files = {"file": (video.name, video.getvalue(), video.type or "video/mp4")}
    with st.status(f"Analyzing {video.name}", expanded=True):
        try:
            response = api_client.post("/api/v1/video/analyze", files=files)
            response.raise_for_status()
            payload = response.json()
            st.write(payload.get("summary"))
            frames = payload.get("frames", [])
            st.dataframe(pd.DataFrame(frames), use_container_width=True)
        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")

st.divider()
query = st.text_input("Search indexed frames")
if st.button("Search frames", disabled=not query):
    st.info("Frame search endpoint is wired as a placeholder until frame embeddings are indexed.")
