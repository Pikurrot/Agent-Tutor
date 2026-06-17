from __future__ import annotations

import logging
import os

import dotenv

dotenv.load_dotenv()

from tutor.utils.config import load_config  # noqa: E402
from tutor.utils.paths import DEFAULT_CONFIG_PATH  # noqa: E402

cfg = load_config(DEFAULT_CONFIG_PATH)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("visible_devices", "0")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

USE_INFERENCE_API = cfg.get("use_inference_api", False)
INFERENCE_API_BASE_URL = str(cfg.get("inference_api_base_url", "http://127.0.0.1:8000")).rstrip("/")

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from tutor.client.inference import StreamingOutcome, iter_streaming_complete, warmup  # noqa: E402
from tutor.core.public_auth import public_auth_configured, verify_public_login  # noqa: E402
from tutor.core.streaming import get_agent_mock_stream_config, mock_stream_text  # noqa: E402
from tutor.core.student_conversation_store import (  # noqa: E402
    StudentConversationStore,
    apply_record_to_session,
    record_from_session,
)
from tutor.modules.agent.pedagogy import TeachingSession  # noqa: E402
from tutor.modules.agent.summarizer import ConversationMemory  # noqa: E402
from tutor.modules.agent.tutor_orchestrator import run_tutor_turn  # noqa: E402
from tutor.modules.retrieval.RAG import RAGModule  # noqa: E402
from tutor.ui.common import (  # noqa: E402
    PUBLIC_API_MODE,
    PUBLIC_MODEL_PATH,
    render_slide_gallery,
    render_status_banner,
)
from tutor.utils.misc import get_model  # noqa: E402
from tutor.utils.paths import MODELS_CACHE_DIR  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Course Tutor", layout="wide")


def _empty_memory_dict() -> dict:
    return {"summary": "", "last_interaction": None}


def _empty_teaching_session_dict() -> dict:
    return TeachingSession.empty().to_dict()


@st.cache_resource
def get_conversation_store() -> StudentConversationStore:
    return StudentConversationStore()


@st.cache_resource
def warmup_resources():
    if USE_INFERENCE_API:
        warmup(INFERENCE_API_BASE_URL, PUBLIC_MODEL_PATH)
        return "api"
    load_model()
    load_rag()
    return "local"


@st.cache_resource
def load_model():
    model_cfg = load_config(DEFAULT_CONFIG_PATH)
    model, model_type = get_model(PUBLIC_MODEL_PATH, str(MODELS_CACHE_DIR), model_cfg)
    model.eval()
    return model, model_type


@st.cache_resource
def load_rag():
    return RAGModule(load_config(DEFAULT_CONFIG_PATH))


def _format_relative_time(iso_ts: str) -> str:
    if not iso_ts:
        return ""
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt.astimezone(timezone.utc)
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        return f"{seconds // 86400}d ago"
    except (TypeError, ValueError):
        return ""


def _persist_active_conversation() -> None:
    store = get_conversation_store()
    record = record_from_session(
        st.session_state.active_conversation_id,
        title=st.session_state.get("conversation_title", "New conversation"),
        created_at=st.session_state.get("conversation_created_at", ""),
        messages=st.session_state.messages,
        conversation_memory=st.session_state.conversation_memory,
        teaching_session=st.session_state.teaching_session,
    )
    store.save(record)
    st.session_state.conversation_title = record.title


def _load_conversation(conversation_id: str) -> None:
    store = get_conversation_store()
    record = store.load(conversation_id)
    session_fields = apply_record_to_session(record)
    for key, value in session_fields.items():
        st.session_state[key] = value


def _ensure_active_conversation() -> None:
    if st.session_state.get("active_conversation_id"):
        return
    store = get_conversation_store()
    summaries = store.list_summaries()
    if summaries:
        _load_conversation(summaries[0].id)
        return
    record = store.create()
    session_fields = apply_record_to_session(record)
    for key, value in session_fields.items():
        st.session_state[key] = value


def _init_session_state() -> None:
    defaults = {
        "active_conversation_id": None,
        "conversation_title": "New conversation",
        "conversation_created_at": "",
        "messages": [],
        "conversation_memory": _empty_memory_dict(),
        "teaching_session": _empty_teaching_session_dict(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_tutor_via_api(prompt: str) -> tuple[str | None, list, dict, dict]:
    outcome = StreamingOutcome()
    with st.container():
        status_slot = st.empty()
        with status_slot.container():
            render_status_banner("Your tutor is preparing a response…")

        def api_token_gen():
            yield from iter_streaming_complete(
                INFERENCE_API_BASE_URL,
                PUBLIC_MODEL_PATH,
                PUBLIC_API_MODE,
                prompt,
                outcome,
                memory=st.session_state.conversation_memory,
                teaching_session=st.session_state.teaching_session,
                debug=False,
            )

        try:
            response_text = st.write_stream(api_token_gen)
        finally:
            status_slot.empty()

    if outcome.text:
        response_text = outcome.text
    memory = outcome.memory or st.session_state.conversation_memory
    session = outcome.teaching_session or st.session_state.teaching_session
    return response_text, outcome.slides or [], memory, session


def _render_login_page() -> None:
    st.title("Course Tutor")
    st.caption("Sign in to continue.")

    if not public_auth_configured():
        st.error(
            "Login is not configured. Set PUBLIC_APP_USERNAME and "
            "PUBLIC_APP_PASSWORD in the server .env file."
        )
        return

    _left, center, _right = st.columns([1, 1, 1])
    with center:
        with st.form("public_login", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if verify_public_login(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")


def _run_tutor_local(prompt: str) -> tuple[str | None, list, dict, dict]:
    model, _ = load_model()
    rag = load_rag()
    current_mem = ConversationMemory.from_dict(st.session_state.conversation_memory)
    current_session = TeachingSession.from_dict(st.session_state.teaching_session)

    status_slot = st.empty()
    with status_slot.container():
        render_status_banner("Searching course material…")

    full_answer, slides, new_mem, new_session, _debug = run_tutor_turn(
        model,
        rag,
        cfg,
        prompt,
        memory=current_mem,
        session=current_session,
        callbacks=None,
        debug=False,
    )

    with status_slot.container():
        render_status_banner("Preparing your tutoring step…")

    stream_cfg = load_config(DEFAULT_CONFIG_PATH)
    delay, chunk = get_agent_mock_stream_config(stream_cfg)
    response = st.write_stream(mock_stream_text(full_answer, delay, chunk))
    status_slot.empty()
    return response, slides or [], new_mem.to_dict(), new_session.to_dict()


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _render_login_page()
    st.stop()

warmup_resources()
_init_session_state()
_ensure_active_conversation()

store = get_conversation_store()
summaries = store.list_summaries()

with st.sidebar:
    st.caption(f"Signed in as {os.environ.get('PUBLIC_APP_USERNAME', 'student')}")
    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()
    st.title("Conversations")
    if st.button("New conversation", use_container_width=True):
        record = store.create()
        session_fields = apply_record_to_session(record)
        for key, value in session_fields.items():
            st.session_state[key] = value
        st.rerun()

    st.divider()
    if not summaries:
        st.caption("No conversations yet.")
    for summary in summaries:
        is_active = summary.id == st.session_state.active_conversation_id
        label = summary.title
        rel = _format_relative_time(summary.updated_at)
        if rel:
            label = f"{summary.title} ({rel})"
        col_open, col_del = st.columns([5, 1])
        with col_open:
            if st.button(
                label,
                key=f"open_{summary.id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if summary.id != st.session_state.active_conversation_id:
                    _load_conversation(summary.id)
                    st.rerun()
        with col_del:
            if st.button("×", key=f"del_{summary.id}", help="Delete conversation"):
                store.delete(summary.id)
                if summary.id == st.session_state.active_conversation_id:
                    remaining = store.list_summaries()
                    if remaining:
                        _load_conversation(remaining[0].id)
                    else:
                        record = store.create()
                        session_fields = apply_record_to_session(record)
                        for key, value in session_fields.items():
                            st.session_state[key] = value
                st.rerun()

st.header("Course Tutor")
st.caption("Ask a question and your tutor will guide you step by step.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_slide_gallery(msg.get("slides"))

if prompt := st.chat_input("Ask your tutor a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    _persist_active_conversation()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text: str | None = None
        slides: list = []
        try:
            if USE_INFERENCE_API:
                response_text, slides, new_memory, new_session = _run_tutor_via_api(prompt)
            else:
                response_text, slides, new_memory, new_session = _run_tutor_local(prompt)

            if slides:
                render_slide_gallery(slides)
            st.session_state.conversation_memory = new_memory
            st.session_state.teaching_session = new_session
        except httpx.HTTPStatusError:
            logger.exception("Inference API HTTP error")
            st.error("The tutor service returned an error. Please try again in a moment.")
            response_text = None
        except httpx.RequestError:
            logger.exception("Inference API unreachable")
            st.error(
                "Could not reach the tutor service. "
                "Make sure it is running (tutor serve), then try again."
            )
            response_text = None
        except Exception:
            logger.exception("Unexpected tutor error")
            st.error("Something went wrong while preparing your answer. Please try again.")
            response_text = None

    if response_text is not None:
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "slides": slides or []}
        )
        _persist_active_conversation()
