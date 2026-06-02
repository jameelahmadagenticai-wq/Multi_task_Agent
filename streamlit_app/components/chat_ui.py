import streamlit as st


def render_trace(history: list[dict]):
    if not history:
        return
    with st.expander("Agent trace", expanded=False):
        for step in history:
            st.write(
                {
                    "agent": step.get("agent"),
                    "status": step.get("status"),
                    "latency_ms": step.get("latency_ms"),
                    "confidence": step.get("confidence"),
                }
            )


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander("Sources", expanded=True):
        st.dataframe(sources, use_container_width=True)
