import pandas as pd
import streamlit as st

import api_client
from components.chat_ui import render_sources
from components.file_uploader import document_uploader
from components.sidebar import render_sidebar


st.set_page_config(page_title="Documents", layout="wide")
render_sidebar()

st.title("Documents")

uploads = document_uploader()
if st.button("Ingest selected files", disabled=not uploads):
    for upload in uploads:
        files = {"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")}
        with st.status(f"Ingesting {upload.name}", expanded=False):
            try:
                response = api_client.post("/api/v1/rag/ingest", files=files)
                response.raise_for_status()
                st.write(response.json())
            except Exception as exc:
                st.error(f"Failed: {exc}")

st.divider()
query = st.text_input("RAG query")
if st.button("Query documents", disabled=not query):
    try:
        response = api_client.post("/api/v1/rag/query", json={"query": query, "top_k": 5})
        response.raise_for_status()
        payload = response.json()
        st.write(payload.get("answer"))
        render_sources(payload.get("sources", []))
    except Exception as exc:
        st.error(f"Query failed: {exc}")

st.divider()
st.subheader("Document Library")
try:
    response = api_client.get("/api/v1/documents")
    response.raise_for_status()
    rows = response.json().get("documents", [])
    st.dataframe(
        pd.DataFrame(rows, columns=["filename", "type", "chunks", "ingested_at", "hash"]),
        use_container_width=True,
        hide_index=True,
    )
except Exception as exc:
    st.info(f"Document library is available when the API and Postgres are running. {exc}")
