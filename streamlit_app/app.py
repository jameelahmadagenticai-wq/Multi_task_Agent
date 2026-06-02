import streamlit as st

from components.sidebar import render_sidebar


st.set_page_config(page_title="Air-Gapped AI", layout="wide")
render_sidebar()

st.title("Air-Gapped Multi-Modal Agentic AI")
st.write("Use the pages in the left navigation to chat, ingest documents, inspect vision inputs, and review memory.")
