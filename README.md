# Agent Tutor

## Setup

1. Install `uv` if not already installed.
2. Run `uv sync` to create the virtual environment and install the package in editable mode.
3. Run `uv run tutor --help` to verify the installation.

# Commands

- `tutor chat`: Run chat with the model.
- `tutor process`: Process a file or directory.
- `tutor app`: Launch Streamlit chat GUI.
- `tutor serve`: Run the inference HTTP API (loads the model and RAG in this process).

## Remote inference (faster Streamlit restarts)

When `use_inference_api` is `true` in [configs/main.yaml](configs/main.yaml), the Streamlit app does not load the model or RAG locally; it sends each request to `inference_api_base_url` (default `http://127.0.0.1:8000`).

1. In one terminal, start the API: `uv run tutor serve` (optional: `--host`, `--port`).
2. Set `use_inference_api: true` in `configs/main.yaml`.
3. In another terminal, run `uv run tutor app` as usual. You can stop and restart the Streamlit process while keeping `tutor serve` running so the heavy weights stay in memory.

The UI streams model output for Basic and RAG (true token streaming). Agent mode streams a **simulated** replay of the final answer; tune delay and chunk size in [configs/agent.yaml](configs/agent.yaml) (`mock_stream_delay_seconds`, `mock_stream_chunk_chars`).
