from __future__ import annotations
import subprocess
import sys

from tutor.utils.paths import PROJECT_ROOT


APP_PATH = PROJECT_ROOT / "src" / "tutor" / "app.py"


def add_app_args(p):
    pass


def run_app(args):
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)])
