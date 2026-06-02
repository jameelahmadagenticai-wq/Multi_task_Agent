import streamlit as st

import api_client
from components.chat_ui import render_sources, render_trace
from components.sidebar import render_sidebar


st.set_page_config(page_title="Chat", layout="wide")
render_sidebar()

st.title("Chat")

mode = st.segmented_control("Mode", ["LLM", "RAG", "Agent"], default="LLM")
task_type = st.selectbox("Task type", ["CHAT", "REASON", "SUMMARIZE", "CODE", "VISION", "AUDIO"])
prompt = st.chat_input("Ask the local AI")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            if mode == "LLM":
                response = api_client.post(
                    "/api/v1/llm/generate",
                    json={
                        "prompt": prompt,
                        "task_type": task_type,
                        "stream": False,
                        "session_id": st.session_state.get("session_id", "default"),
                    },
                )
                response.raise_for_status()
                st.write(response.json()["answer"])

            elif mode == "RAG":
                response = api_client.post("/api/v1/rag/query", json={"query": prompt, "top_k": 5})
                response.raise_for_status()
                payload = response.json()
                st.write(payload.get("answer"))
                render_sources(payload.get("sources", []))

            else:
                response = api_client.post(
                    "/api/v1/agent/run",
                    json={
                        "query": prompt,
                        "context": {},
                        "session_id": st.session_state.get("session_id", "default"),
                    },
                )
                response.raise_for_status()
                payload = response.json()
                st.write(payload.get("final_answer") or payload.get("results"))
                render_trace(payload.get("history", []))

        except Exception as exc:
            st.error(f"Request failed: {exc}")
