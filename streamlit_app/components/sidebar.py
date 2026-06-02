import streamlit as st

import api_client


def render_sidebar():
    st.sidebar.title("Air-Gapped AI")
    st.sidebar.caption("Local models, RAG, agents, memory")

    st.sidebar.divider()
    st.sidebar.text_input("Session", key="session_id", value=st.session_state.get("session_id", "default"))

    if st.sidebar.button("New session", use_container_width=True):
        try:
            api_client.delete("/api/v1/memory/session", session_id=st.session_state.session_id)
            st.toast("Session cleared")
        except Exception as exc:
            st.sidebar.error(f"Could not clear session: {exc}")

    render_recent_conversation()
    render_long_term_memory_search()


def render_recent_conversation():
    st.sidebar.subheader("Recent Conversation")
    try:
        response = api_client.get("/api/v1/memory/recent", session_id=st.session_state.get("session_id", "default"))
        if response.status_code != 200:
            st.sidebar.caption("No recent memory available.")
            return
        history = response.json()
    except Exception:
        st.sidebar.caption("Memory service unavailable.")
        return

    if not history:
        st.sidebar.caption("No turns in this session yet.")
        return

    for item in history[-10:]:
        label = f"{item.get('role', 'turn')} - {item.get('agent_used') or 'chat'}"
        with st.sidebar.expander(label):
            st.write(item.get("content", ""))


def render_long_term_memory_search():
    st.sidebar.subheader("Long-Term Memory")
    query = st.sidebar.text_input("Search memories", key="memory_query")
    if st.sidebar.button("Search memory", use_container_width=True, disabled=not query):
        try:
            response = api_client.post("/api/v1/memory/search", json={"query": query, "top_k": 5})
            response.raise_for_status()
            for memory in response.json().get("results", []):
                with st.sidebar.expander(f"Score {memory.get('importance_score', 0):.2f}"):
                    st.write(memory.get("content", ""))
        except Exception as exc:
            st.sidebar.error(f"Search failed: {exc}")
