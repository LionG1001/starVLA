# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .qwen35_musa import configure_qwen35_musa_fla_path
from .qwen35_musa_flash_attention import (
    configure_qwen35_musa_vision_flash_attention,
    resolve_qwen35_attention_implementation,
)
from .qwen35_musa_vision_patch import configure_qwen35_musa_vision_patch_path

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 248077 + 2047 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


def _config_bool(value: Any, *, name: str) -> bool:
    """Parse a strict bool so quoted YAML/CLI values cannot silently invert it."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, but got {value!r}.")


def _configure_gradient_checkpointing(model: nn.Module, trainer_config: Any) -> bool:
    """Apply and verify the Qwen3.5 activation-checkpointing policy."""
    setting_name = "trainer.enable_gradient_checkpointing"
    enabled = _config_bool(
        trainer_config.get("enable_gradient_checkpointing", False),
        name=setting_name,
    )
    configure = getattr(
        model,
        "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable",
        None,
    )
    if not callable(configure):
        raise RuntimeError(
            f"{setting_name}={enabled} was requested, but this Qwen3.5 model "
            "does not expose the corresponding gradient-checkpointing API."
        )
    configure()
    active = bool(getattr(model, "is_gradient_checkpointing", False))
    if active != enabled:
        raise RuntimeError(
            f"Failed to apply {setting_name}={enabled}: "
            f"model.is_gradient_checkpointing={active}."
        )
    return active


def _accelerator_autocast(dtype: torch.dtype):
    """Select MUSA or CUDA autocast without misrouting the device string."""
    if hasattr(torch, "musa") and torch.musa.is_available():
        return torch.autocast(device_type="musa", dtype=dtype)
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _sdpa_backend_context(attn_implementation: str, sdpa_backend: str):
    """Optionally restrict SDPA to a known backend for this model call."""
    if attn_implementation != "sdpa" or sdpa_backend == "auto":
        return nullcontext()
    if sdpa_backend == "math":
        return sdpa_kernel(SDPBackend.MATH)
    if sdpa_backend == "flash":
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    raise ValueError(
        f"Unsupported SDPA backend {sdpa_backend!r}; expected 'auto', 'math', or 'flash'."
    )


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-4B

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-4B")
        requested_attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        attn_implementation = resolve_qwen35_attention_implementation(
            requested_attn_implementation
        )
        sdpa_backend = qwenvl_config.get("sdpa_backend", "auto")

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        self.gradient_checkpointing_enabled = _configure_gradient_checkpointing(
            model,
            config.trainer,
        )
        self.musa_vision_flash_attention_layers = (
            configure_qwen35_musa_vision_flash_attention(model, qwenvl_config)
        )
        self.musa_vision_patch_linear_layers = (
            configure_qwen35_musa_vision_patch_path(model, qwenvl_config)
        )
        self.musa_fla_linear_layers = configure_qwen35_musa_fla_path(
            model, qwenvl_config
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self.attn_implementation = attn_implementation
        self.requested_attn_implementation = requested_attn_implementation
        self.sdpa_backend = sdpa_backend

        # Align the composite Qwen3.5 config with older VLM wrappers.
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        # QwenOFT trains from hidden states rather than autoregressive decoding:
        # disable the KV cache and retain logits only for the final token to
        # avoid allocating the full [batch, sequence, vocabulary] tensor.
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("logits_to_keep", 1)
        with _sdpa_backend_context(
            self.attn_implementation, self.sdpa_backend
        ), _accelerator_autocast(torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with _sdpa_backend_context(
            self.attn_implementation, self.sdpa_backend
        ), _accelerator_autocast(torch.bfloat16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow the official Qwen3.5 format: https://huggingface.co/Qwen/Qwen3.5-4B
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)




if __name__ == "__main__":
    import argparse

    import debugpy
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./models/Qwen3.5-4B"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
