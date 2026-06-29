from __future__ import annotations
import random
import os
import numpy as np
import torch
import json
from transformers import Qwen2_5_VLConfig
from typing import Tuple, Any

from tutor.modules.models.gemini import GeminiModel
from tutor.modules.models.openai_model import OpenAIModel
from tutor.modules.models.qwen import QwenVLModel, Qwen
from tutor.modules.models.groq import GroqModel


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def split_string(string: str, delimeters: list[str]) -> list[str]:
    if not delimeters:
        return [string.strip()]

    d = delimeters[0]
    parts = string.split(d)
    result = []
    for part in parts:
        result.extend(split_string(part, delimeters[1:]))
    return [x.strip() for x in result if x.strip()]

def parse_json_response(response: str) -> dict:
	response = response.strip()
	if response.startswith("```json"):
		response = response[len("```json"):].strip()
		if response.endswith("```"):
			response = response[:-3].strip()
	elif response.startswith("```"):
		response = response[3:].strip()
		if response.endswith("```"):
			response = response[:-3].strip()
	try:
		response_dict = json.loads(response)
	except json.JSONDecodeError:
		start = response.find("{")
		end = response.rfind("}")
		if start != -1 and end != -1 and start < end:
			response_dict = json.loads(response[start:end+1])
		else:
			raise
	return response_dict

def get_model(model_path: str, cache_dir: str, config: dict) -> Tuple[Any, str]:
	config["model_path"] = model_path
	if model_path.startswith("gemini"):
		model_type = "gemini"
		model = GeminiModel(config)
	elif model_path.startswith("gpt-"):
		model_type = "openai"
		model = OpenAIModel(config)
	elif model_path.startswith("groq"):
		model_type = "groq"
		model = GroqModel(config)
	else:
		model_type = "qwen"
		if "VL" in model_path:
			qwen_config = Qwen2_5_VLConfig.from_pretrained(
				model_path,
				cache_dir=cache_dir,
				torch_dtype=torch.bfloat16,
				attn_implementation=None#"flash_attention_2" if IS_151 else None
			)
			qwen_config.update(config.get("qwen_config", {}))
			model = QwenVLModel(
				model_path,
				cache_dir=cache_dir,
				config=qwen_config,
			)
		else:
			from transformers import AutoConfig
			# Check if it's a LoRA adapter
			is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
			if is_lora:
				import json
				with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
					adapter_config = json.load(f)
				base_model_path = adapter_config.get("base_model_name_or_path")
				print(f"Detected LoRA adapter. Loading base model from {base_model_path}")
				actual_model_path = base_model_path
				config["qwen_config"]["lora_weights"] = model_path
			else:
				actual_model_path = model_path

			qwen_config = AutoConfig.from_pretrained(
				actual_model_path,
				cache_dir=cache_dir,
				trust_remote_code=True
			)
			# Update with qwen_config from dict
			for k, v in config.get("qwen_config", {}).items():
				setattr(qwen_config, k, v)
			
			model = Qwen(
				actual_model_path,
				cache_dir=cache_dir,
				config=qwen_config,
			)
	return model, model_type
