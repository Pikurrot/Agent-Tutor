from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from tutor.server.inference import iter_completion, run_completion
from tutor.server.schemas import CompleteRequest, CompleteResponse, SlideOut

app = FastAPI(title="Agent Tutor Inference")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/complete", response_model=CompleteResponse)
def complete(body: CompleteRequest):
    try:
        text, slide_dicts = run_completion(body.model_path, body.mode, body.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    slides = [SlideOut(**d) for d in slide_dicts]
    return CompleteResponse(text=text, slides=slides)


@app.post("/v1/complete/stream")
def complete_stream(body: CompleteRequest):
    def ndjson():
        try:
            for kind, data in iter_completion(body.model_path, body.mode, body.prompt):
                if kind == "token":
                    line = json.dumps({"t": "tok", "d": data}, ensure_ascii=False) + "\n"
                    yield line.encode("utf-8")
                elif kind == "slides":
                    line = json.dumps({"t": "end", "slides": data}, ensure_ascii=False) + "\n"
                    yield line.encode("utf-8")
        except Exception as e:
            err = json.dumps({"t": "err", "d": str(e)}, ensure_ascii=False) + "\n"
            yield err.encode("utf-8")

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")
