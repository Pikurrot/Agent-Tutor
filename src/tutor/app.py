from __future__ import annotations
import os
import dotenv
import traceback

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

from tutor.client.inference import StreamingOutcome, iter_streaming_complete  # noqa: E402
from tutor.core.streaming import get_agent_mock_stream_config, mock_stream_text  # noqa: E402
from tutor.utils.misc import get_model  # noqa: E402
from tutor.utils.paths import MODELS_CACHE_DIR  # noqa: E402
from tutor.core.chat import stream_generate_response  # noqa: E402
from tutor.modules.retrieval.RAG import RAGModule  # noqa: E402
from tutor.modules.agent.agent import build_rag_agent, StreamlitAgentCallbackHandler  # noqa: E402
from tutor.modules.agent.pedagogy import TeachingSession  # noqa: E402
from tutor.modules.agent.summarizer import ConversationMemory, roll_memory  # noqa: E402
from tutor.modules.agent.tutor_orchestrator import run_tutor_turn  # noqa: E402
from tutor.ui.common import render_slide_gallery  # noqa: E402


def _empty_memory_dict() -> dict:
    return {"summary": "", "last_interaction": None}


def _empty_teaching_session_dict() -> dict:
    return TeachingSession.empty().to_dict()

ANSWER_MODE_TO_API = {"Basic": "basic", "RAG": "rag", "Agent": "agent", "Tutor": "tutor"}
STATEFUL_MODES = {"Agent", "Tutor"}


def render_debug_panel(debug_data: dict | None) -> None:
    if not debug_data:
        return
    with st.expander("Debug trace", expanded=False):
        tab_overview, tab_anchor, tab_answer, tab_pedagogic = st.tabs(
            ["Overview", "Teaching Anchor", "Answer Agent", "Pedagogic Agent"]
        )
        with tab_overview:
            col1, col2 = st.columns(2)
            col1.metric("Session status", str(debug_data.get("session_status", "n/a")))
            col2.metric("Hints used", int(debug_data.get("hints_used", 0)))
            st.caption("Per-turn internal diagnostics for tutor development.")
        with tab_anchor:
            st.json(debug_data.get("teaching_anchor_snapshot"), expanded=False)
        with tab_answer:
            st.json(debug_data.get("answer_agent_trace"), expanded=False)
        with tab_pedagogic:
            st.json(debug_data.get("pedagogic_agent_trace"), expanded=False)


MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Groq": "groq/openai/gpt-oss-120b",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-VL-8B": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
}

st.set_page_config(page_title="Agent Tutor", layout="centered")


@st.cache_resource
def load_model(model_path: str):
    model_cfg = load_config(DEFAULT_CONFIG_PATH)
    model, model_type = get_model(model_path, str(MODELS_CACHE_DIR), model_cfg)
    model.eval()
    return model, model_type


@st.cache_resource
def load_rag():
    return RAGModule(load_config(DEFAULT_CONFIG_PATH))


# Sidebar
with st.sidebar:
    st.title("Settings")
    selected_model = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[selected_model]
    answer_mode = st.selectbox(
        "Answer mode",
        ["Basic", "RAG", "Agent", "Tutor"],
        index=0,
        help="Basic: your message is sent to the model as-is. RAG: lecture context is retrieved and prepended first. Agent: use a ReAct agent to answer the question. Tutor: Socratic dual-agent that guides you toward the answer instead of revealing it.",
    )
    if USE_INFERENCE_API:
        st.caption(f"Remote inference: `{INFERENCE_API_BASE_URL}`")
    debug_mode = st.toggle(
        "Debug mode",
        value=False,
        help="Show internal tutor traces (teaching anchor + both agent reasonings).",
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_memory = _empty_memory_dict()
        st.session_state.teaching_session = _empty_teaching_session_dict()
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = _empty_memory_dict()
if "teaching_session" not in st.session_state:
    st.session_state.teaching_session = _empty_teaching_session_dict()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_slide_gallery(msg.get("slides"))
            if bool(debug_mode) and msg.get("debug_data") is not None:
                render_debug_panel(msg.get("debug_data"))

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if USE_INFERENCE_API:
        api_mode = ANSWER_MODE_TO_API[answer_mode]
        response_text: str | None = None
        slides: list | None = None
        with st.chat_message("assistant"):
            outcome = StreamingOutcome()
            memory_to_send = (
                st.session_state.conversation_memory if answer_mode in STATEFUL_MODES else None
            )
            teaching_session_to_send = (
                st.session_state.teaching_session if answer_mode == "Tutor" else None
            )

            def api_token_gen():
                yield from iter_streaming_complete(
                    INFERENCE_API_BASE_URL,
                    model_path,
                    api_mode,
                    prompt,
                    outcome,
                    memory=memory_to_send,
                    teaching_session=teaching_session_to_send,
                    debug=bool(debug_mode and answer_mode == "Tutor"),
                )

            try:
                response_text = st.write_stream(api_token_gen)
            except httpx.HTTPStatusError as e:
                detail = e.response.text
                st.error(f"Inference API error ({e.response.status_code}): {detail}")
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
                response_text = None
            except httpx.RequestError as e:
                st.error(
                    f"Could not reach inference API at {INFERENCE_API_BASE_URL}. "
                    f"Start it with: tutor serve. ({e})"
                )
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
                response_text = None
            except RuntimeError as e:
                st.error(f"Inference error: {e}")
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
                response_text = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
                response_text = None
            else:
                if outcome.text:
                    response_text = outcome.text
                slides = outcome.slides
                if answer_mode in ("RAG", "Agent", "Tutor") and slides:
                    render_slide_gallery(slides)
                if answer_mode in STATEFUL_MODES and outcome.memory is not None:
                    st.session_state.conversation_memory = outcome.memory
                if answer_mode == "Tutor" and outcome.teaching_session is not None:
                    st.session_state.teaching_session = outcome.teaching_session
                if bool(debug_mode) and answer_mode == "Tutor" and outcome.debug_data is not None:
                    render_debug_panel(outcome.debug_data)
        if response_text is not None:
            assistant_entry: dict = {"role": "assistant", "content": response_text}
            if answer_mode in ("RAG", "Agent", "Tutor"):
                assistant_entry["slides"] = slides or []
            if answer_mode == "Tutor":
                assistant_entry["debug_data"] = outcome.debug_data
            st.session_state.messages.append(assistant_entry)
    else:
        model, model_type = load_model(model_path)

        with st.chat_message("assistant"):
            try:
                if answer_mode == "RAG":
                    with st.spinner("Loading RAG..."):
                        rag = load_rag()
                    with st.status("Retrieving context...", expanded=False) as status:

                        def report(msg: str) -> None:
                            status.update(label=msg, state="running")
                            status.write(msg)

                        model_prompt, slides, rag_images = rag.retrieve_and_augment(prompt, on_progress=report)
                        status.update(label="Retrieval complete", state="complete", expanded=False)
                    response = st.write_stream(stream_generate_response(model, model_prompt, images=rag_images))
                    render_slide_gallery(slides)
                elif answer_mode == "Agent":
                    with st.spinner("Initializing agent..."):
                        rag = load_rag()
                        current_mem = ConversationMemory.from_dict(
                            st.session_state.conversation_memory
                        )
                        agent_executor, slide_manager = build_rag_agent(
                            model, rag, cfg, memory=current_mem
                        )
                    with st.status("Thinking and Retrieving...", expanded=False) as status:
                        def report(msg: str) -> None:
                            status.update(label=msg, state="running")
                            status.write(msg)

                        slide_manager.set_progress_callback(report)
                        st_callback = StreamlitAgentCallbackHandler(status)
                        response_dict = agent_executor.invoke(
                            {"input": prompt},
                            config={"callbacks": [st_callback]},
                        )
                        full_answer = response_dict["output"]
                        slides = slide_manager.retrieved_slides
                        status.update(label="Response generated", state="complete", expanded=False)

                    stream_cfg = load_config(DEFAULT_CONFIG_PATH)
                    delay, chunk = get_agent_mock_stream_config(stream_cfg)
                    response = st.write_stream(
                        mock_stream_text(full_answer, delay, chunk)
                    )
                    if slides:
                        render_slide_gallery(slides)

                    agent_cfg = cfg.get("agent_config", {}) or {}
                    memory_cfg = agent_cfg.get("memory", {}) or {}
                    if bool(memory_cfg.get("enabled", True)):
                        with st.spinner("Updating conversation memory..."):
                            new_mem = roll_memory(
                                current_mem,
                                new_user=prompt,
                                new_assistant=full_answer,
                                model=model,
                                summary_max_new_tokens=int(
                                    memory_cfg.get("summary_max_new_tokens", 512)
                                ),
                            )
                        st.session_state.conversation_memory = new_mem.to_dict()
                elif answer_mode == "Tutor":
                    with st.spinner("Initializing Socratic tutor..."):
                        rag = load_rag()
                        current_mem = ConversationMemory.from_dict(
                            st.session_state.conversation_memory
                        )
                        current_session = TeachingSession.from_dict(
                            st.session_state.teaching_session
                        )
                    with st.status("Thinking and Retrieving...", expanded=False) as status:
                        def report(msg: str) -> None:
                            status.update(label=msg, state="running")
                            status.write(msg)

                        st_callback = StreamlitAgentCallbackHandler(status)
                        full_answer, slides, new_mem, new_session, debug_data = run_tutor_turn(
                            model,
                            rag,
                            cfg,
                            prompt,
                            memory=current_mem,
                            session=current_session,
                            callbacks=[st_callback],
                            debug=bool(debug_mode),
                        )
                        status.update(label="Tutoring move ready", state="complete", expanded=False)

                    stream_cfg = load_config(DEFAULT_CONFIG_PATH)
                    delay, chunk = get_agent_mock_stream_config(stream_cfg)
                    response = st.write_stream(
                        mock_stream_text(full_answer, delay, chunk)
                    )
                    if slides:
                        render_slide_gallery(slides)
                    if bool(debug_mode) and debug_data is not None:
                        render_debug_panel(debug_data)

                    st.session_state.conversation_memory = new_mem.to_dict()
                    st.session_state.teaching_session = new_session.to_dict()
                else:
                    model_prompt = prompt
                    slides = None
                    response = st.write_stream(stream_generate_response(model, model_prompt))
            except Exception as e:
                st.error(f"An error occurred during local inference: {e}")
                st.exception(e)
                st.code(traceback.format_exc(), language="python")
                response = None
                slides = None

        assistant_entry = {"role": "assistant", "content": response}
        if answer_mode in ("RAG", "Agent", "Tutor"):
            assistant_entry["slides"] = slides
        if answer_mode == "Tutor":
            assistant_entry["debug_data"] = (
                debug_data if "debug_data" in locals() else None
            )
        st.session_state.messages.append(assistant_entry)
