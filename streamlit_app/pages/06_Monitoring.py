import pandas as pd
import streamlit as st

import api_client
from components.sidebar import render_sidebar


st.set_page_config(page_title="Monitoring", layout="wide")
render_sidebar()

st.title("Monitoring")

try:
    health = api_client.get("/health")
    api_status = "ok" if health.ok else f"error {health.status_code}"
except Exception as exc:
    api_status = f"offline: {exc}"

st.metric("API", api_status)

st.subheader("LLMOps")
st.info("Prometheus metrics are exposed by FastAPI at /metrics and scraped by the Docker stack.")

st.subheader("Evaluation")
eval_rows = [
    {"suite": "hallucination", "status": "available", "script": "eval/hallucination.py"},
    {"suite": "retrieval_quality", "status": "available", "script": "eval/retrieval_quality.py"},
    {"suite": "agent_accuracy", "status": "available", "script": "eval/agent_accuracy.py"},
]
st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
