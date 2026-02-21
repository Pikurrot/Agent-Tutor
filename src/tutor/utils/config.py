from __future__ import annotations
from pathlib import Path
from typing import Union, Dict, Any

from tutor.utils.file_handler import read_yaml


EXCLUDED_CONFIG_FILES = []

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively updates a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(path: Union[Path, str]) -> dict:
    """
    Loads configuration from a file or a directory of YAML files.
    If a directory is provided, all .yaml and .yml files are loaded and merged.
    """
    path = Path(path)
    if path.is_file():
        return read_yaml(path)
    
    if path.is_dir():
        merged_config = {}
        # Order matters if there are overrides, but here we assume logical separation
        # We'll load main.yaml first if it exists, then others
        main_cfg_path = path / "main.yaml"
        if main_cfg_path.exists():
            merged_config = read_yaml(main_cfg_path)
        
        for cfg_file in sorted(path.glob("*.yaml")):
            if cfg_file.name == "main.yaml" or cfg_file.name in EXCLUDED_CONFIG_FILES:
                continue
            cfg_data = read_yaml(cfg_file)
            if cfg_data:
                deep_update(merged_config, cfg_data)
        
        for cfg_file in sorted(path.glob("*.yml")):
            if cfg_file.name in EXCLUDED_CONFIG_FILES:
                continue
            cfg_data = read_yaml(cfg_file)
            if cfg_data:
                deep_update(merged_config, cfg_data)
                
        return merged_config
    
    raise FileNotFoundError(f"Config path {path} not found.")
