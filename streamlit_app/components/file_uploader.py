import streamlit as st


def document_uploader():
    return st.file_uploader(
        "Documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
    )


def image_uploader(label="Image"):
    return st.file_uploader(label, type=["png", "jpg", "jpeg", "webp"])
