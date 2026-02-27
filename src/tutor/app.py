from __future__ import annotations
import os
import dotenv

dotenv.load_dotenv()

from tutor.utils.config import load_config # noqa: E402
from tutor.utils.paths import DEFAULT_CONFIG_PATH # noqa: E402

cfg = load_config(DEFAULT_CONFIG_PATH)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("visible_devices", "0")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st  # noqa: E402

from tutor.utils.misc import get_model  # noqa: E402
from tutor.utils.paths import MODELS_CACHE_DIR  # noqa: E402
from tutor.core.chat import stream_generate_response  # noqa: E402

MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Groq": "groq/openai/gpt-oss-120b",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

st.set_page_config(page_title="Agent Tutor", layout="centered")


@st.cache_resource
def load_model(model_path: str):
    model_cfg = load_config(DEFAULT_CONFIG_PATH)
    model, model_type = get_model(model_path, str(MODELS_CACHE_DIR), model_cfg)
    model.eval()
    return model, model_type


# Sidebar
with st.sidebar:
    st.title("Settings")
    selected_model = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[selected_model]

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    model, model_type = load_model(model_path)

    with st.chat_message("assistant"):
        response = st.write_stream(stream_generate_response(model, prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
