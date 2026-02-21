from __future__ import annotations
import os
import yaml
import json
import numpy as np
from pathlib import Path

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json(path: str, data: dict, **kwargs):
    smart = kwargs.get("smart", False)
    smart_start_level = kwargs.get("smart_start_level", 1)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w+") as f:
        if smart:
            f.write(smart_json_dumps(data, indent=4, smart_start_level=smart_start_level, max_inline_length=1000))
        else:
            json.dump(data, f, indent=4, cls=NpEncoder)

def save_yaml(path: str, data: dict):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w+") as f:
        yaml.dump(data, f)

def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def is_simple(obj):
    """
    Returns True if the object is simple enough to be dumped in a single line.
    'Simple' means:
    - It is a primitive (str, int, float, bool, None)
    - It is a list whose items are all simple
    - It is a dict whose keys (assumed to be strings) and values are all simple
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return True
    elif isinstance(obj, list):
        return all(is_simple(item) for item in obj)
    elif isinstance(obj, dict):
        return all(isinstance(k, str) and is_simple(v) for k, v in obj.items())
    return False

def smart_json_dumps(
    obj,
    indent=2,
    level=0,
    max_inline_length=80,
    smart_start_level=0,
    ignore_keys=["centroid"]
):
    """
    Recursively dumps a JSON object with smart formatting.
    
    The parameter `smart_start_level` defines the level from which smart formatting is applied:
    - If the current recursion level is less than `smart_start_level`, the object is formatted in
        a normal (always expanded) multi-line way.
    - If the current level is >= smart_start_level, then:
            - If the object is "simple" and its inline JSON representation is not too long,
            it is printed inline.
            - Otherwise, it is formatted with newlines and proper indentation.
    
    :param obj: The Python object to dump.
    :param indent: Number of spaces to use for each indentation level.
    :param level: The current recursion level (used internally).
    :param max_inline_length: Maximum length of inline JSON string for it to be kept on one line.
    :param smart_start_level: The recursion level at which smart inline formatting begins.
    """
    current_indent = ' ' * (indent * level)
    next_indent = ' ' * (indent * (level + 1))
    
    # When we haven't reached the smart level, format normally (always expanded)
    if level < smart_start_level:
        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                if isinstance(k, int) or isinstance(k, float):
                    key_str = json.dumps(str(k), ensure_ascii=False, cls=NpEncoder)
                else:
                    key_str = json.dumps(k, ensure_ascii=False, cls=NpEncoder)

                # Recursively format children; they might be smart if level+1 >= smart_start_level.
                value_str = smart_json_dumps(v, indent, level + 1, max_inline_length, smart_start_level)
                items.append(f"{next_indent}{key_str}: {value_str}")
            return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"
        
        elif isinstance(obj, list):
            if not obj:
                return '[]'
            items = []
            for item in obj:
                item_str = smart_json_dumps(item, indent, level + 1, max_inline_length, smart_start_level)
                items.append(f"{next_indent}{item_str}")
            return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"
        
        else:
            return json.dumps(obj, ensure_ascii=False, cls=NpEncoder)
    
    # At levels where smart formatting applies:
    if is_simple(obj):
        inline_repr = json.dumps(obj, ensure_ascii=False, cls=NpEncoder)
        if len(inline_repr) <= max_inline_length:
            return inline_repr

    if isinstance(obj, dict):
        if not obj:
            return '{}'
        items = []
        for k, v in obj.items():
            if k in ignore_keys:
                continue
            if isinstance(k, int):
                key_str = json.dumps(str(k), ensure_ascii=False, cls=NpEncoder)
            else:
                key_str = json.dumps(k, ensure_ascii=False, cls=NpEncoder)

            value_str = smart_json_dumps(v, indent, level + 1, max_inline_length, smart_start_level)
            items.append(f"{next_indent}{key_str}: {value_str}")
        return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"
    
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        items = []
        for item in obj:
            item_str = smart_json_dumps(item, indent, level + 1, max_inline_length, smart_start_level)
            items.append(f"{next_indent}{item_str}")
        return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"
    
    # For any other types, fall back to json.dumps.
    return json.dumps(obj, ensure_ascii=False, cls=NpEncoder)

def prepare_alternative_output_path(
    results_path: Path,
    suffix: str
) -> Path:
    results_dir = results_path.parent
    output_dir = results_dir.parent / f"{results_dir.name}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_filename = results_path.name
    output_path = output_dir / results_filename
    return output_path
