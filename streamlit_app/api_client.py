import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def get_token():
    if "access_token" not in st.session_state:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/token",
            json={"username": "local_user", "role": "user"},
            timeout=30,
        )
        response.raise_for_status()
        st.session_state.access_token = response.json()["access_token"]
    return st.session_state.access_token


def headers():
    return {"Authorization": f"Bearer {get_token()}"}


def get(path: str, **params):
    return requests.get(f"{API_BASE_URL}{path}", headers=headers(), params=params, timeout=120)


def post(path: str, json=None, files=None, data=None, stream=False):
    return requests.post(
        f"{API_BASE_URL}{path}",
        headers=headers(),
        json=json,
        files=files,
        data=data,
        stream=stream,
        timeout=600,
    )


def delete(path: str, **params):
    return requests.delete(f"{API_BASE_URL}{path}", headers=headers(), params=params, timeout=120)
