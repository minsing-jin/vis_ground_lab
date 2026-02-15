"""Florence-2 model wrapper with optional LoRA adapter training."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from vis_ground_lab.base import BaseVGModel, BoundingBox, VGSample


class Florence2Wrapper(BaseVGModel):
    """HF-based Florence-2 wrapper implementing the BaseVGModel interface."""

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str | Mapping[str, Any] = "auto",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = ("q_proj", "v_proj"),
    ) -> None:
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.device_map = device_map

        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = list(target_modules)

        self.processor: AutoProcessor | None = None
        self.model: AutoModelForCausalLM | PeftModel | None = None

    def load_model(self, adapter_path_or_repo: str | None = None, is_trainable_adapter: bool = False) -> None:
        """Load processor/model and optionally attach a LoRA adapter."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

        if adapter_path_or_repo is not None:
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path_or_repo,
                is_trainable=is_trainable_adapter,
            )
        elif self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=self.target_modules,
            )
            model = get_peft_model(base_model, lora_config)
            model.print_trainable_parameters()
        else:
            model = base_model

        self.model = model

    def preprocess(self, sample: VGSample) -> Mapping[str, Any]:
        """Convert one sample into processor tensors ready for model input."""
        if self.processor is None:
            raise RuntimeError("Processor is not loaded. Call load_model() first.")

        return self.processor(
            text=sample.text,
            images=sample.image,
            return_tensors="pt",
        )

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Forward pass through Florence-2."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        outputs = self.model(**batch)
        return dict(outputs.items())

    def predict(self, image: Any, text: str) -> BoundingBox:
        """Generate output text and parse a single bbox from the response."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model and processor must be loaded first via load_model().")

        inputs = self.processor(text=text, images=image, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        return self._parse_bbox(generated_text)

    def push_to_hub(self, token: str, repo_name: str) -> None:
        """Upload model artifacts to Hugging Face Hub."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        self.model.push_to_hub(repo_name, token=token)
        if self.processor is not None:
            self.processor.push_to_hub(repo_name, token=token)

    @classmethod
    def from_pretrained_adapter(
        cls,
        base_model_name: str,
        adapter_path_or_repo: str,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str | Mapping[str, Any] = "auto",
    ) -> "Florence2Wrapper":
        """Create wrapper from base model + trained LoRA adapter (local path or HF repo)."""
        wrapper = cls(
            model_name=base_model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_lora=False,
        )
        wrapper.load_model(adapter_path_or_repo=adapter_path_or_repo, is_trainable_adapter=False)
        return wrapper

    @staticmethod
    def _parse_bbox(text: str) -> BoundingBox:
        """Parse the first 4 numeric tokens as bbox coordinates."""
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(numbers) < 4:
            raise ValueError(f"Unable to parse bounding box from model output: {text}")

        x1, y1, x2, y2 = map(float, numbers[:4])
        return BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
