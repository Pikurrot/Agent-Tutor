from __future__ import annotations

import subprocess
import sys

from tutor.utils.paths import PROJECT_ROOT

PUBLIC_APP_PATH = PROJECT_ROOT / "src" / "tutor" / "app_public.py"


def add_public_args(p):
    p.add_argument(
        "--port",
        type=int,
        default=8502,
        help="Streamlit server port (default: 8502)",
    )


def run_public(args):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PUBLIC_APP_PATH),
            "--server.port",
            str(args.port),
        ]
    )
