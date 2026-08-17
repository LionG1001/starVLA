# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import socket
import time
from collections import deque
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.mfu import (
    QWEN35_MFU_FORMULA_VERSION,
    Qwen35BatchFlopShape,
    Qwen35ModelFlopConfig,
    estimate_qwen35_training_flops,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args


def _configure_tf32_from_env() -> None:
    """Apply an explicit TF32 policy before Accelerate initializes each rank."""
    setting = os.getenv("STARVLA_ALLOW_TF32", "auto").strip().lower()
    if setting == "auto":
        return
    if setting in {"1", "true", "yes", "on"}:
        enabled = True
    elif setting in {"0", "false", "no", "off"}:
        enabled = False
    else:
        raise ValueError(
            "STARVLA_ALLOW_TF32 must be auto/1/0/true/false/on/off, "
            f"but got {setting!r}"
        )

    # torch_musa's muBLAS path uses the CUDA-compatible matmul flag, while
    # muDNN exposes its own flag. Set both so the policy is unambiguous.
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = enabled
    if hasattr(torch.backends, "mudnn"):
        torch.backends.mudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    print(
        "StarVLA TF32 policy: "
        f"allow={enabled}, "
        f"cuda_matmul={torch.backends.cuda.matmul.allow_tf32}, "
        f"mudnn={getattr(getattr(torch.backends, 'mudnn', None), 'allow_tf32', 'n/a')}, "
        f"matmul_precision={torch.get_float32_matmul_precision()}",
        flush=True,
    )


_configure_tf32_from_env()

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _parse_bool_flag(value, *, name: str) -> bool:
    """Parse config/environment boolean flags without treating "false" as true."""
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
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _build_qwen35_flop_config(model) -> Qwen35ModelFlopConfig | None:
    """Collect Qwen3.5 component sizes once, before DeepSpeed wraps the model."""
    qwen_interface = getattr(model, "qwen_vl_interface", None)
    backbone = getattr(qwen_interface, "model", None)
    backbone_config = getattr(backbone, "config", None)
    if getattr(backbone_config, "model_type", None) != "qwen3_5":
        return None

    try:
        text_config = backbone_config.text_config
        vision_config = backbone_config.vision_config
        language_model = backbone.model.language_model
        vision_model = backbone.model.visual
        layer_types = list(text_config.layer_types)
        full_attention_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
        linear_attention_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
        if len(layer_types) != text_config.num_hidden_layers or full_attention_layers == 0:
            raise ValueError(
                "Qwen3.5 layer_types must describe every layer and contain full-attention layers"
            )

        return Qwen35ModelFlopConfig(
            text_layer_parameters=_parameter_count(language_model.layers),
            full_attention_layers=full_attention_layers,
            linear_attention_layers=linear_attention_layers,
            text_attention_heads=int(text_config.num_attention_heads),
            text_head_dim=int(text_config.head_dim),
            linear_num_value_heads=int(text_config.linear_num_value_heads),
            linear_key_head_dim=int(text_config.linear_key_head_dim),
            linear_value_head_dim=int(text_config.linear_value_head_dim),
            vision_block_parameters=_parameter_count(vision_model.blocks),
            vision_patch_parameters=_parameter_count(vision_model.patch_embed),
            vision_merger_parameters=_parameter_count(vision_model.merger),
            vision_layers=int(vision_config.depth),
            vision_attention_heads=int(vision_config.num_heads),
            vision_head_dim=int(vision_config.hidden_size // vision_config.num_heads),
            vision_spatial_merge_size=int(vision_config.spatial_merge_size),
            action_parameters=_parameter_count(model.action_model),
            lm_head_parameters=_parameter_count(backbone.lm_head),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Failed to collect Qwen3.5 MFU model metadata") from exc


def _build_adamw_optimizer(param_groups, cfg):
    """Build AdamW, using torch_musa's fused implementation when requested."""
    fused_requested = _parse_bool_flag(
        cfg.trainer.optimizer.get("fused", False),
        name="trainer.optimizer.fused",
    )
    env_override = os.getenv("STARVLA_ENABLE_FUSED_OPTIMIZER")
    if env_override is not None:
        fused_requested = _parse_bool_flag(
            env_override,
            name="STARVLA_ENABLE_FUSED_OPTIMIZER",
        )

    optimizer_class = torch.optim.AdamW
    fused_enabled = False
    if fused_requested:
        musa_available = hasattr(torch, "musa") and torch.musa.is_available()
        if musa_available:
            try:
                # Import the submodule explicitly: ``import torch_musa`` alone
                # does not expose ``torch_musa.optim`` in every release.
                from torch_musa.optim import FusedAdamW

                optimizer_class = FusedAdamW
                fused_enabled = True
            except (ImportError, AttributeError) as exc:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.warning(
                        "MUSA FusedAdamW was requested but is unavailable; "
                        f"falling back to torch.optim.AdamW: {exc}"
                    )
        elif not dist.is_initialized() or dist.get_rank() == 0:
            logger.warning(
                "FusedAdamW was requested outside a MUSA runtime; "
                "falling back to torch.optim.AdamW"
            )

    optimizer = optimizer_class(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "Optimizer implementation: "
            f"{optimizer_class.__module__}.{optimizer_class.__name__} "
            f"(fused_requested={fused_requested}, fused_enabled={fused_enabled})"
        )

    return optimizer


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = _build_adamw_optimizer(param_groups=param_groups, cfg=cfg)

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # TFLOPs and MFU tracking - GPU Peak TFLOPs for MUSA GPU
        # This is an explicit per-device BF16 peak assumption, not a runtime query.
        self.gpu_peak_tflops = float(getattr(cfg.trainer, "gpu_peak_tflops", 460.0))
        if self.gpu_peak_tflops <= 0:
            raise ValueError("trainer.gpu_peak_tflops must be positive")
        self.qwen35_flop_config = None
        self.mfu_formula_version = "legacy_qwen25_v1"
        self.mfu_warmup_steps = int(getattr(cfg.trainer, "mfu_warmup_steps", 10))
        self.mfu_window_size = int(getattr(cfg.trainer, "mfu_window_size", 20))
        if self.mfu_warmup_steps < 0:
            raise ValueError("trainer.mfu_warmup_steps must be non-negative")
        if self.mfu_window_size <= 0:
            raise ValueError("trainer.mfu_window_size must be positive")
        self._mfu_window = deque(maxlen=self.mfu_window_size)

    def _create_profiler(self):
        """Create an opt-in, bounded profiler for the selected distributed ranks."""
        enabled = _parse_bool_flag(
            os.getenv("STARVLA_PROFILE_ENABLED", "0"),
            name="STARVLA_PROFILE_ENABLED",
        )
        if not enabled:
            return None

        rank = self.accelerator.process_index
        rank_spec = os.getenv("STARVLA_PROFILE_RANKS", "0").strip().lower()
        if rank_spec not in {"all", "*"}:
            try:
                selected_ranks = {int(value.strip()) for value in rank_spec.split(",") if value.strip()}
            except ValueError as exc:
                raise ValueError(
                    "STARVLA_PROFILE_RANKS must be 'all' or a comma-separated list of integer ranks"
                ) from exc
            if rank not in selected_ranks:
                return None

        def _schedule_value(name: str, default: int, minimum: int = 0) -> int:
            value = int(os.getenv(name, str(default)))
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
            return value

        wait_steps = _schedule_value("STARVLA_PROFILE_WAIT", 2)
        warmup_steps = _schedule_value("STARVLA_PROFILE_WARMUP", 1)
        active_steps = _schedule_value("STARVLA_PROFILE_ACTIVE", 3, minimum=1)
        repeat = _schedule_value("STARVLA_PROFILE_REPEAT", 1, minimum=1)

        trace_root = Path(
            os.getenv(
                "STARVLA_PROFILE_DIR",
                os.path.join(self.config.output_dir, "traces"),
            )
        )
        rank_trace_dir = trace_root / f"rank_{rank:02d}"
        rank_trace_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if hasattr(torch, "musa") and torch.musa.is_available():
            activities.append(torch.profiler.ProfilerActivity.MUSA)
        elif torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        use_gzip = _parse_bool_flag(
            os.getenv("STARVLA_PROFILE_GZIP", "1"),
            name="STARVLA_PROFILE_GZIP",
        )
        profile_memory = _parse_bool_flag(
            os.getenv("STARVLA_PROFILE_MEMORY", "1"),
            name="STARVLA_PROFILE_MEMORY",
        )
        with_stack = _parse_bool_flag(
            os.getenv("STARVLA_PROFILE_STACK", "1"),
            name="STARVLA_PROFILE_STACK",
        )

        metadata = {
            "rank": rank,
            "world_size": self.accelerator.num_processes,
            "host": socket.gethostname(),
            "torch_version": torch.__version__,
            "activities": [activity.name for activity in activities],
            "schedule": {
                "wait": wait_steps,
                "warmup": warmup_steps,
                "active": active_steps,
                "repeat": repeat,
            },
            "record_shapes": True,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
        }
        (rank_trace_dir / "profile_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Profiler enabled on rank %s: wait=%s warmup=%s active=%s repeat=%s output=%s",
            rank,
            wait_steps,
            warmup_steps,
            active_steps,
            repeat,
            rank_trace_dir,
        )
        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(rank_trace_dir),
                worker_name=f"{socket.gethostname()}_rank{rank:02d}",
                use_gzip=use_gzip,
            ),
            record_shapes=True,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        self.qwen35_flop_config = _build_qwen35_flop_config(self.model)
        if self.qwen35_flop_config is not None:
            self.mfu_formula_version = QWEN35_MFU_FORMULA_VERSION
        def nan_check_hook(module, inputs, output):
            def has_bad(x):
                return torch.isnan(x).any() or torch.isinf(x).any()

            outs = output if isinstance(output, tuple) else (output,)

            for out in outs:
                if isinstance(out, torch.Tensor) and has_bad(out):
                    print(f"\n[NaN in {module.__class__.__name__}]")

                    # 打印输入
                    for i, inp in enumerate(inputs):
                        if isinstance(inp, torch.Tensor):
                            print(f" input[{i}]: shape={inp.shape}, "
                                f"min={inp.min().item()}, max={inp.max().item()}")

                    # 打印输出
                    print(f" output: shape={out.shape}")

                    raise RuntimeError("NaN detected")

        # 注册（挑重点模块更高效，比如 attention / mlp）
        handles = []
        # for name, m in self.model.named_modules():
        #     print(name, m)
        #     handles.append(m.register_forward_hook(nan_check_hook))
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        # self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if dist.get_rank() != 0:
            return

        if "estimated_tflops_per_device_step" in metrics and "model_time" in metrics:
            step_tflops = float(metrics["estimated_tflops_per_device_step"])
            step_time = float(metrics["model_time"])
            if self.completed_steps > self.mfu_warmup_steps and step_time > 0:
                self._mfu_window.append((step_tflops, step_time))

        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

            # Calculate per-device MFU from useful model FLOPs and wall time.
            if "estimated_tflops_per_device_step" in metrics and "model_time" in metrics:
                model_tflops = metrics["estimated_tflops_per_device_step"]
                model_time = metrics["model_time"]
                achieved_tflops = model_tflops / model_time if model_time > 0 else 0
                mfu_percent = (achieved_tflops / self.gpu_peak_tflops) * 100
                metrics["achieved_tflops_per_device"] = achieved_tflops
                metrics["peak_tflops_per_device"] = self.gpu_peak_tflops
                metrics["mfu_percent"] = mfu_percent
                if self._mfu_window:
                    window_tflops = sum(item[0] for item in self._mfu_window)
                    window_time = sum(item[1] for item in self._mfu_window)
                    rolling_tflops = window_tflops / window_time
                    metrics["achieved_tflops_per_device_rolling"] = rolling_tflops
                    metrics["mfu_percent_rolling"] = (
                        rolling_tflops / self.gpu_peak_tflops * 100
                    )
                    metrics["mfu_window_steps"] = len(self._mfu_window)

            log_msg = f"Step {self.completed_steps}, Loss: {metrics}"
            # Keep one unwrapped, machine-readable line in the launcher log.
            print(log_msg, flush=True)

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )
        profiler = self._create_profiler()
        if profiler is not None:
            profiler.start()

        try:
            while self.completed_steps < self.config.trainer.max_train_steps:
                t_start_data = time.perf_counter()
                with torch.profiler.record_function("starvla.data_loading"):
                    batch_vla = self._get_next_batch()
                t_end_data = time.perf_counter()

                t_start_model = time.perf_counter()
                with torch.profiler.record_function("starvla.train_step"):
                    step_metrics = self._train_step(batch_vla)
                t_end_model = time.perf_counter()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.completed_steps += 1

                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    with torch.profiler.record_function("starvla.evaluation"):
                        step_metrics = self.eval_action_model(step_metrics)

                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    with torch.profiler.record_function("starvla.checkpoint"):
                        self._save_checkpoint()

                if profiler is not None:
                    profiler.step()
        finally:
            if profiler is not None:
                profiler.stop()

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  GPU Peak TFLOPs per device = {self.gpu_peak_tflops}")
            logger.info(f"  MFU formula = {self.mfu_formula_version}")
            logger.info(
                "  MFU rolling window = %s measured steps after %s warmup steps",
                self.mfu_window_size,
                self.mfu_warmup_steps,
            )
            logger.info("  MFU scope = useful per-device model FLOPs; checkpoint recomputation excluded")

    def _estimate_legacy_model_tflops(self, batch_size, seq_len=2048):
        """
        Estimate model TFLOPs for one forward pass.
        For Qwen2.5-VL-3B model + action head.
        """
        # Get model config
        hidden_size = getattr(self.model.qwen_vl_interface.model.config, 'hidden_size', 2560)
        num_layers = getattr(self.model.qwen_vl_interface.model.config, 'num_hidden_layers', 36)
        intermediate_size = getattr(self.model.qwen_vl_interface.model.config, 'intermediate_size', 9728)
        vocab_size = getattr(self.model.qwen_vl_interface.model.config, 'vocab_size', 151936)

        # Vision component FLOPs (approximation)
        # Assuming 336x336 images, patch size 14, num_patches = 576
        num_patches = 576  # For 336x336 with patch_size=14
        vision_flops = 2 * num_patches * hidden_size * num_layers * 4  # Rough approximation

        # Transformer FLOPs per layer (forward pass)
        # Attention: 4 * batch * seq_len^2 * hidden_size
        # MLP: 2 * batch * seq_len * hidden_size * intermediate_size * 2
        attention_flops = 4 * batch_size * seq_len * hidden_size * hidden_size
        mlp_flops = 4 * batch_size * seq_len * hidden_size * intermediate_size
        layer_flops = attention_flops + mlp_flops

        # Total transformer FLOPs
        transformer_flops = num_layers * layer_flops
        lm_head = vocab_size*2*hidden_size*batch_size*seq_len
        # Action head FLOPs (small MLP)
        action_dim = getattr(self.model, 'chunk_len', 16) * 7  # chunk_len * action_dim
        action_hidden = getattr(self.model.action_model, 'action_hidden_dim', 512)
        action_flops = 2 * batch_size * action_hidden * action_dim * 3  # 3 layer MLP

        # Total FLOPs (forward pass)
        total_flops = transformer_flops + vision_flops + action_flops + lm_head

        # Forward + backward = 3x forward FLOPs (approximation)
        # For mixed precision training with bfloat16
        training_flops_per_step = total_flops * 3

        return training_flops_per_step / 1e12  # Convert to TFLOPs

    def _estimate_model_tflops(self, batch_size, mfu_metadata=None):
        """Return per-device FLOPs metrics for the active model and batch."""
        if self.qwen35_flop_config is None:
            return {
                "estimated_tflops_per_device_step": self._estimate_legacy_model_tflops(batch_size),
                "mfu_formula": self.mfu_formula_version,
            }

        if mfu_metadata is None:
            raise RuntimeError("Qwen3.5 forward did not return MFU batch metadata")
        shape = Qwen35BatchFlopShape(
            batch_size=int(mfu_metadata["batch_size"]),
            padded_language_sequence_length=int(mfu_metadata["padded_language_sequence_length"]),
            vision_patch_tokens=int(mfu_metadata["vision_patch_tokens"]),
            num_images=int(mfu_metadata["num_images"]),
            action_tokens_per_sample=int(mfu_metadata["action_tokens_per_sample"]),
            logits_tokens_per_sample=int(mfu_metadata.get("logits_tokens_per_sample", 1)),
        )
        estimate = estimate_qwen35_training_flops(self.qwen35_flop_config, shape)
        return {
            "estimated_tflops_per_device_step": estimate["estimated_tflops_per_device_step"],
            "estimated_text_tflops_per_device_step": estimate[
                "estimated_text_tflops_per_device_step"
            ],
            "estimated_vision_tflops_per_device_step": estimate[
                "estimated_vision_tflops_per_device_step"
            ],
            "estimated_action_tflops_per_device_step": estimate[
                "estimated_action_tflops_per_device_step"
            ],
            "mfu_formula": estimate["formula_version"],
            "language_tokens_per_device": estimate["language_tokens_per_device"],
            "vision_patch_tokens_per_device": estimate["vision_patch_tokens_per_device"],
            "vision_merged_tokens_per_device": estimate["vision_merged_tokens_per_device"],
            "action_tokens_per_device": estimate["action_tokens_per_device"],
        }

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        batch_size = len(batch_vla) if isinstance(batch_vla, list) else batch_vla.get('input_ids', torch.zeros(1)).shape[0]

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # MUSA support: use cuda autocast for both CUDA and MUSA (torch_musa compatible)
            with torch.profiler.record_function("starvla.forward"):
                with torch.autocast("musa", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = action_loss

            with torch.profiler.record_function("starvla.backward"):
                self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                with torch.profiler.record_function("starvla.gradient_clipping"):
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            with torch.profiler.record_function("starvla.optimizer"):
                self.optimizer.step()
                self.lr_scheduler.step()

        flop_metrics = self._estimate_model_tflops(
            batch_size,
            mfu_metadata=output_dict.get("mfu_metadata"),
        )
        return {
            "action_dit_loss": action_loss.item(),
            **flop_metrics,
        }

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # if self.accelerator.is_main_process:
        #     wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
