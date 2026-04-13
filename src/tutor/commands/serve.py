from __future__ import annotations

import uvicorn


def add_serve_args(p):
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Listen port (default: 8000)",
    )


def run_serve(args):
    uvicorn.run(
        "tutor.server.app:app",
        host=args.host,
        port=args.port,
    )
