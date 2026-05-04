from __future__ import annotations
import threading
import torch
from transformers import (
	Qwen3VLForConditionalGeneration,
	Qwen3VLConfig,
	AutoProcessor,
	AutoModelForCausalLM,
	AutoTokenizer,
	TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
from peft import PeftModel
from typing import Tuple, Optional, Any, Generator, List
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

from tutor.utils.paths import MODELS_CACHE_DIR
from tutor.modules.models.utils import get_generative_confidence
from tutor.modules.models.base import BaseModel


class QwenVLModel(torch.nn.Module, BaseModel):
	def __init__(self, model_path: str, cache_dir: str, config: Qwen3VLConfig):
		super(QwenVLModel, self).__init__()
		self.lora_weights = config.lora_weights
		self.processor = AutoProcessor.from_pretrained(model_path)
		device_map = getattr(config, "device_map", "auto")
		max_memory = getattr(config, "max_memory", None)
		self.model = Qwen3VLForConditionalGeneration.from_pretrained(
			model_path,
			config=config,
			cache_dir=str(MODELS_CACHE_DIR),
			torch_dtype=config.torch_dtype,
			attn_implementation=config._attn_implementation,
			device_map=device_map,
			max_memory=max_memory,
		)
		if self.lora_weights:
			print(f"Loading LoRA weights from {self.lora_weights}")
			self.model = PeftModel.from_pretrained(
				self.model,
				self.lora_weights,
				torch_dtype=config.torch_dtype,
			)
			print("LoRA weights loaded successfully")
		if getattr(config, "gradient_checkpointing", False):
			self.model.gradient_checkpointing_enable()
			self.model.config.use_cache = False

		first_device = next(self.model.parameters()).device
		self.device = first_device
		self.max_seq_length = config.max_seq_length
		self.max_new_tokens = config.max_new_tokens
		self.train_mode = False
		self.qwen_downsize_images = config.qwen_downsize_images
		self.image_max_size = config.image_max_size

	def to(self, device):
		pass

	def eval(self):
		self.model.eval()
		self.train_mode = False

	def train(self):
		self.model.train()
		self.train_mode = True

	def generate(
		self,
		prompt: str,
		images: Optional[list] = None,
		stop: Optional[List[str]] = None,
		max_new_tokens: Optional[int] = None,
		**kwargs,
	) -> str:
		images_list = [images] if images else None
		inputs, _, _ = self.prepare_inputs_for_vqa([prompt], images_list)
		pred_answers, _ = self.get_answer_from_model_output(
			inputs, stop=stop, max_new_tokens=max_new_tokens
		)
		return pred_answers[0]

	def stream_generate(self, prompt: str, images: Optional[list] = None, **kwargs) -> Generator[str, None, None]:
		images_list = [images] if images else None
		inputs, _, _ = self.prepare_inputs_for_vqa([prompt], images_list)
		streamer = TextIteratorStreamer(
			self.processor.tokenizer, skip_special_tokens=True, skip_prompt=True,
		)
		gen_kwargs = {**inputs, "max_new_tokens": self.max_new_tokens, "streamer": streamer}
		thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
		thread.start()
		for text in streamer:
			if text:
				yield text
		thread.join()

	def prepare_inputs_for_vqa(
			self,
			prompts: list,      # (bs,) context strings
			images: Optional[list] = None,    # (bs, k) PIL images
			answers: Optional[list] = None  # (bs,) Optional ground truth answers
	) -> dict:
		if isinstance(prompts, str):
			prompts = [prompts]
		if images is None:
			images = []
		resized_images = []
		if not self.qwen_downsize_images:
			for batch_imgs in images:
				batch_resized = []
				for img in batch_imgs:
					if img.width < 28 or img.height < 28:
						new_width = max(img.width, 28)
						new_height = max(img.height, 28)
						batch_resized.append(img.resize((new_width, new_height)))
					else:
						batch_resized.append(img)
				resized_images.append(batch_resized)
		else:
			for batch_imgs in images:
				batch_resized = []
				for img in batch_imgs:
					max_size = self.image_max_size
					if img.width < 28 or img.height < 28:
						new_width = max(img.width, 28)
						new_height = max(img.height, 28)
						batch_resized.append(img.resize((new_width, new_height)))
					elif img.width > max_size or img.height > max_size:
						aspect = img.width / img.height
						if aspect > 1:
							new_width = max_size
							new_height = int(max_size / aspect)
						else:
							new_height = max_size
							new_width = int(max_size * aspect)
						batch_resized.append(img.resize((new_width, new_height), Image.LANCZOS))
					else:
						batch_resized.append(img)
				resized_images.append(batch_resized)

		messages = []
		for i in range(len(prompts)):
			message = {
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": prompts[i]
					}
				]
			}
			if resized_images:
				for img in resized_images[i]:
					message["content"].append({
						"type": "image",
						"image": img,
					})
			messages.append([message])

		prompts = [
			self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
			for msg in messages
		]
		image_inputs, _ = process_vision_info(messages)

		inputs = self.processor(
			text=prompts,
			images=image_inputs,
			videos=None,
			padding=True,
			return_tensors="pt",
			padding_side="left",
		)
		
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		
		labels = None
		if answers:
			messages_with_answers = []
			for i in range(len(prompts)):
				msg = [
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": prompts[i]
							}
						]
					}
				]
				for img in resized_images[i]:
					msg[0]["content"].append({
						"type": "image",
						"image": img,
					})
				msg.append({
					"role": "assistant",
					"content": answers[i]
				})
				messages_with_answers.append(msg)
			
			combined_prompts = [
				self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
				for msg in messages_with_answers
			]
			image_inputs_combined, _ = process_vision_info(messages_with_answers)
			
			combined_inputs = self.processor(
				text=combined_prompts,
				images=image_inputs_combined,
				videos=None,
				padding=True,
				return_tensors="pt",
				padding_side="left",
			)
			
			combined_inputs = {k: v.to(self.device) for k, v in combined_inputs.items()}
			
			labels = combined_inputs["input_ids"].clone()
			prompt_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
			
			for i in range(len(labels)):
				labels[i, :prompt_lengths[i]] = -100
			
			return inputs, combined_inputs, labels
		
		return inputs, None, None

	def forward(
			self,
			prompts: list,
			images: Optional[list] = None,
			answers: Optional[list] = None,
			return_pred_answer: bool = False
	):
		"""
		Forward pass that directly accepts prompts and images instead of a batch dictionary.
		
		Args:
			prompts: List of prompt strings
			images: List of lists of PIL images
			answers: Optional list of ground truth answers (for training)
			return_pred_answer: Whether to return predicted answers
			
		Returns:
			outputs: Model outputs (if in training mode)
			pred_answers: Predicted answers (if return_pred_answer=True or in inference mode)
			pred_answers_conf: Confidence scores for predictions
		"""
		answers_to_use = answers if self.train_mode else None
		
		inputs, combined_inputs, labels = self.prepare_inputs_for_vqa(prompts, images, answers_to_use)

		if labels is not None:
			outputs = self.model(
				**combined_inputs,
				labels=labels
			)

			if return_pred_answer:
				pred_answers, pred_answers_conf = self.get_answer_from_model_output(inputs)
			else:
				pred_answers, pred_answers_conf = None, None
		else:
			outputs = None
			pred_answers, pred_answers_conf = self.get_answer_from_model_output(inputs)

		return outputs, pred_answers, pred_answers_conf

	def get_answer_from_model_output(
			self,
			inputs: dict,
			stop: Optional[List[str]] = None,
			max_new_tokens: Optional[int] = None,
	) -> Tuple[list, list]:
		gen_kwargs: dict = {
			**inputs,
			"output_scores": True,
			"return_dict_in_generate": True,
			"output_attentions": False,
			"max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
		}
		if stop:
			gen_kwargs["stop_strings"] = list(stop)
			gen_kwargs["tokenizer"] = self.processor.tokenizer
		with torch.no_grad():
			output = self.model.generate(**gen_kwargs)

		generated_ids_trimmed = [
			out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output.sequences)
		]

		pred_answers = self.processor.batch_decode(
			generated_ids_trimmed, 
			skip_special_tokens=True, 
			clean_up_tokenization_spaces=False
		)

		pred_answers_conf = get_generative_confidence(output)

		return pred_answers, pred_answers_conf


class Qwen(torch.nn.Module, BaseModel):
	def __init__(self, model_path: str, cache_dir: str, config: Any):
		super(Qwen, self).__init__()
		self.lora_weights = getattr(config, "lora_weights", None)
		self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		device_map = getattr(config, "device_map", "auto")
		max_memory = getattr(config, "max_memory", None)
		
		self.model = AutoModelForCausalLM.from_pretrained(
			model_path,
			cache_dir=cache_dir,
			torch_dtype=getattr(config, "torch_dtype", "auto"),
			device_map=device_map,
			max_memory=max_memory,
		)
		if self.lora_weights:
			print(f"Loading LoRA weights from {self.lora_weights}")
			self.model = PeftModel.from_pretrained(
				self.model,
				self.lora_weights,
				torch_dtype=getattr(config, "torch_dtype", "auto"),
			)
			print("LoRA weights loaded successfully")
		if getattr(config, "gradient_checkpointing", False):
			self.model.gradient_checkpointing_enable()
			self.model.config.use_cache = False

		self.device = next(self.model.parameters()).device
		self.max_new_tokens = getattr(config, "max_new_tokens", 32768)
		self.train_mode = False

	def to(self, device):
		pass

	def eval(self):
		self.model.eval()
		self.train_mode = False

	def train(self):
		self.model.train()
		self.train_mode = True

	def generate(
		self,
		prompt: str,
		stop: Optional[List[str]] = None,
		max_new_tokens: Optional[int] = None,
		**kwargs,
	) -> str:
		inputs, _, _ = self.prepare_inputs([prompt])
		pred_answers, _ = self.get_answer_from_model_output(
			inputs, stop=stop, max_new_tokens=max_new_tokens
		)
		return pred_answers[0]

	def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
		inputs, _, _ = self.prepare_inputs([prompt])
		streamer = TextIteratorStreamer(
			self.tokenizer, skip_special_tokens=True, skip_prompt=True,
		)
		gen_kwargs = {**inputs, "max_new_tokens": self.max_new_tokens, "streamer": streamer}
		thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
		thread.start()
		for text in streamer:
			if text:
				yield text
		thread.join()

	def prepare_inputs(self, prompts: list, answers: Optional[list] = None) -> Tuple[dict, Optional[dict], Optional[torch.Tensor]]:
		if isinstance(prompts, str):
			prompts = [prompts]
		
		messages = [[{"role": "user", "content": p}] for p in prompts]

		templated_prompts = [
			self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
			for msg in messages
		]

		inputs = self.tokenizer(
			templated_prompts,
			padding=True,
			return_tensors="pt",
			padding_side="left"
		).to(self.device)

		if answers:
			messages_with_answers = []
			for p, a in zip(prompts, answers):
				messages_with_answers.append([
					{"role": "user", "content": p},
					{"role": "assistant", "content": a}
				])
			
			combined_prompts = [
				self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False, enable_thinking=False)
				for msg in messages_with_answers
			]

			combined_inputs = self.tokenizer(
				combined_prompts,
				padding=True,
				return_tensors="pt",
				padding_side="left"
			).to(self.device)

			labels = combined_inputs["input_ids"].clone()
			prompt_lengths = [len(ids) for ids in inputs["input_ids"]]
			for i in range(len(labels)):
				labels[i, :prompt_lengths[i]] = -100
			
			return inputs, combined_inputs, labels
		
		return inputs, None, None

	def forward(
			self,
			prompts: list,
			images: Optional[list] = None,
			answers: Optional[list] = None,
			return_pred_answer: bool = False
	):
		answers_to_use = answers if self.train_mode else None
		inputs, combined_inputs, labels = self.prepare_inputs(prompts, answers_to_use)

		if labels is not None:
			outputs = self.model(
				**combined_inputs,
				labels=labels
			)
			if return_pred_answer:
				pred_answers, pred_answers_conf = self.get_answer_from_model_output(inputs)
			else:
				pred_answers, pred_answers_conf = None, None
		else:
			outputs = None
			pred_answers, pred_answers_conf = self.get_answer_from_model_output(inputs)

		return outputs, pred_answers, pred_answers_conf

	def get_answer_from_model_output(
			self,
			inputs: dict,
			stop: Optional[List[str]] = None,
			max_new_tokens: Optional[int] = None,
	) -> Tuple[list, list]:
		gen_kwargs: dict = {
			**inputs,
			"output_scores": True,
			"return_dict_in_generate": True,
			"max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
		}
		if stop:
			gen_kwargs["stop_strings"] = list(stop)
			gen_kwargs["tokenizer"] = self.tokenizer
		with torch.no_grad():
			output = self.model.generate(**gen_kwargs)

		generated_ids_trimmed = [
			out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output.sequences)
		]

		pred_answers = self.tokenizer.batch_decode(
			generated_ids_trimmed,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=False
		)

		pred_answers_conf = get_generative_confidence(output)
		return pred_answers, pred_answers_conf


class LangChainQwen(LLM):  
    _model: Any = PrivateAttr() # avoid Pydantic trying to validate the PyTorch module
    _slide_manager: Any = PrivateAttr(default=None)
    _agent_max_new_tokens: int = PrivateAttr(default=1024)

    def __init__(
        self,
        qwen_model: BaseModel,
        slide_manager: Any = None,
        agent_max_new_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model: BaseModel = qwen_model
        self._slide_manager: Any = slide_manager
        self._agent_max_new_tokens: int = agent_max_new_tokens

    def _call(
		self,
		prompt: str,
		stop: Optional[List[str]] = None,
		**kwargs: Any
	) -> str:
        response = self._model.generate(
            prompt,
            images=None,
            stop=stop,
            max_new_tokens=self._agent_max_new_tokens,
        )

        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response[:response.index(stop_token)]

        return response

    @property
    def _llm_type(self) -> str:
        return "custom_qwen"
