"""Florence-2 model wrapper with optional LoRA adapter training."""

from __future__ import annotations

import re
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    BartTokenizer,
    BartTokenizerFast,
)

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
        cache_dir: str = ".hf_cache",
        train_image_size: int = 384,
        train_image_seq_length: int = 256,
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
        self.cache_dir = str(Path(cache_dir))
        self.train_image_size = train_image_size
        self.train_image_seq_length = train_image_seq_length

        self.processor: Any | None = None
        self.model: AutoModelForCausalLM | PeftModel | None = None

    def load_model(self, adapter_path_or_repo: str | None = None, is_trainable_adapter: bool = False) -> None:
        """Load processor/model and optionally attach a LoRA adapter."""
        self._configure_hf_cache_paths()
        runtime_dtype = self.torch_dtype
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            if runtime_dtype in (torch.float16, torch.bfloat16):
                runtime_dtype = torch.float32
        config: Any | None = None
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=False,
                cache_dir=self.cache_dir,
            )
            # Florence-2 remote modeling can fail on sdpa init checks in some
            # transformers/torch combinations. Force eager attention.
            setattr(config, "_attn_implementation", "eager")
        except OSError:
            config = None
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=False,
                cache_dir=self.cache_dir,
            )
        except Exception:
            try:
                tokenizer = BartTokenizerFast.from_pretrained(
                    self.model_name,
                    trust_remote_code=False,
                    cache_dir=self.cache_dir,
                )
            except Exception:
                tokenizer = BartTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=False,
                    cache_dir=self.cache_dir,
                )

            # Florence-2 processor expects image token metadata on tokenizer.
            if not hasattr(tokenizer, "image_token"):
                tokenizer.image_token = "<image>"
            if not hasattr(tokenizer, "image_token_id"):
                tokenizer.image_token_id = int(getattr(config, "image_token_id", -1))

            image_processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=False,
                cache_dir=self.cache_dir,
            )
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=False,
                    cache_dir=self.cache_dir,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                )
            except Exception:
                self.processor = _FallbackVLProcessor(image_processor=image_processor, tokenizer=tokenizer)

        if config is None:
            config = getattr(self.processor, "config", None) or SimpleNamespace(image_token_id=-1)

        image_token_id = self._synchronize_image_token(config)
        # Training-time processor overrides can break inference-time generation
        # shape assumptions, so only apply them when adapters are trainable.
        if self.use_lora or is_trainable_adapter:
            self._configure_processor_for_training()

        model_load_kwargs = dict(
            trust_remote_code=self.trust_remote_code,
            dtype=runtime_dtype,
            device_map=self.device_map,
            cache_dir=self.cache_dir,
        )
        if config is not None:
            model_load_kwargs["config"] = config
            model_load_kwargs["attn_implementation"] = "eager"

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_load_kwargs,
            )
        except ImportError as exc:
            message = str(exc)
            if "einops" in message or "timm" in message:
                raise ImportError(
                    "Florence-2 requires extra packages. Install with: "
                    "`pip install einops timm` (or reinstall myvg after pulling latest pyproject)."
                ) from exc
            raise
        except OSError as exc:
            raise OSError(
                f"Failed to load model weights for '{self.model_name}'. "
                f"Check internet access or pre-download model files into cache_dir='{self.cache_dir}'."
            ) from exc

        # Keep token embedding size aligned when we add `<image>` special token.
        if (
            self.processor is not None
            and hasattr(self.processor, "tokenizer")
            and hasattr(base_model, "get_input_embeddings")
            and hasattr(base_model, "resize_token_embeddings")
        ):
            tokenizer = self.processor.tokenizer
            embeddings = base_model.get_input_embeddings()
            if embeddings is not None and image_token_id >= embeddings.weight.shape[0]:
                base_model.resize_token_embeddings(len(tokenizer))

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

    def _synchronize_image_token(self, config: Any) -> int:
        """Ensure processor/tokenizer agree on a valid `<image>` token id."""
        if self.processor is None or not hasattr(self.processor, "tokenizer"):
            return int(getattr(config, "image_token_id", -1))

        tokenizer = self.processor.tokenizer
        image_token = "<image>"

        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or image_token_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        if hasattr(self.processor, "image_token"):
            self.processor.image_token = image_token
        if hasattr(self.processor, "image_token_id"):
            self.processor.image_token_id = int(image_token_id)
        setattr(config, "image_token_id", int(image_token_id))
        return int(image_token_id)

    def _configure_hf_cache_paths(self) -> None:
        """Route HF cache writes to a workspace-local directory."""
        cache_root = Path(self.cache_dir).resolve()
        modules_cache = cache_root / "modules"
        cache_root.mkdir(parents=True, exist_ok=True)
        modules_cache.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("HF_HOME", str(cache_root))
        os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
        try:
            import transformers.dynamic_module_utils as dmu

            dmu.HF_MODULES_CACHE = str(modules_cache)
        except Exception:
            pass

    def _configure_processor_for_training(self) -> None:
        """Reduce sequence pressure so Florence-2 fits within text positional limits."""
        if self.processor is None:
            return
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            image_processor.size = {"height": self.train_image_size, "width": self.train_image_size}
            image_processor.crop_size = {"height": self.train_image_size, "width": self.train_image_size}
        if hasattr(self.processor, "image_seq_length"):
            self.processor.image_seq_length = int(self.train_image_seq_length)

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


class _FallbackVLProcessor:
    """Fallback processor when Florence-2 remote AutoProcessor init fails."""

    def __init__(self, image_processor: Any, tokenizer: Any) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Any,
        images: Any,
        return_tensors: str = "pt",
        padding: bool | str = False,
        truncation: bool = False,
    ) -> Mapping[str, Any]:
        tokenized = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
        )
        vision = self.image_processor(images=images, return_tensors=return_tensors)
        merged = dict(tokenized)
        merged.update(dict(vision))
        return merged

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def push_to_hub(self, repo_name: str, token: str) -> None:
        self.tokenizer.push_to_hub(repo_name, token=token)
        self.image_processor.push_to_hub(repo_name, token=token)
