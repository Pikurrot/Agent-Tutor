from __future__ import annotations
from pathlib import Path


# Main Roots
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAVE_ROOT = PROJECT_ROOT / "save"
MODELS_CACHE_DIR = Path("/data/151-1/users/elopez/models")

# Configs
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs"
