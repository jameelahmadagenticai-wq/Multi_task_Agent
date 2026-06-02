import streamlit as st

import api_client
from components.sidebar import render_sidebar


st.set_page_config(page_title="Voice", layout="wide")
render_sidebar()

st.title("Voice")

session_id = st.session_state.get("session_id", "default")
audio = st.file_uploader("Audio file", type=["wav", "mp3", "ogg", "m4a"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Transcribe", disabled=not audio):
        files = {"file": (audio.name, audio.getvalue(), audio.type or "audio/wav")}
        try:
            response = api_client.post("/api/v1/audio/transcribe", files=files)
            response.raise_for_status()
            payload = response.json()
            st.text_area("Transcript", payload.get("text", ""), height=160)
            if payload.get("error"):
                st.warning(payload["error"])
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")

with col2:
    if st.button("Run command", disabled=not audio):
        files = {"file": (audio.name, audio.getvalue(), audio.type or "audio/wav")}
        try:
            response = api_client.post(
                "/api/v1/audio/command",
                files=files,
                data={"session_id": session_id},
            )
            response.raise_for_status()
            payload = response.json()
            st.text_area("Transcript", payload.get("transcript", ""), height=120)
            st.json(
                {
                    "intent": payload.get("intent"),
                    "entities": payload.get("entities"),
                    "confidence": payload.get("confidence"),
                }
            )
            st.write(payload.get("answer"))
        except Exception as exc:
            st.error(f"Voice command failed: {exc}")
