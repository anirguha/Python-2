# save_load_models_utils.py

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


# ============================================================
# 1. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    save_path: str | Path,
    model: nn.Module,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
    heads_class_names: Optional[Dict[str, List[str]]] = None,
    img_size: Optional[int] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    best_metric: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a rich checkpoint containing:
      - model state_dict
      - architecture info (module, class name, kwargs)
      - class names (single head or multiple heads)
      - training state: optimizer, scheduler, scaler, epoch, global_step, best_metric
      - extra_metadata: anything else you want

    Args:
        save_path: where to save (.pth)
        model: trained model (nn.Module)
        model_kwargs: kwargs needed to reconstruct the model (num_classes, image_size, etc.)
        class_names: list of class names for single-head models
        heads_class_names: dict of {head_name: [class names]} for multi-head models
        img_size: input image size if relevant (e.g. 224, 384)
        optimizer: optional optimizer whose state will be saved
        scheduler: optional LR scheduler whose state will be saved
        scaler: optional GradScaler (for AMP) whose state will be saved
        epoch: current epoch (for resume)
        global_step: global training step (for resume)
        best_metric: best validation metric so far (for resume)
        extra_metadata: any extra info you want (dict)
    """
    save_path = _to_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Architecture info
    arch_info = {
        "module": model.__class__.__module__,   # e.g. "torchvision.models.vision_transformer"
        "class_name": model.__class__.__name__, # e.g. "VisionTransformer"
        "kwargs": model_kwargs or {},           # user-provided kwargs
    }

    # Class info
    class_info: Dict[str, Any] = {}
    if class_names is not None:
        class_info["class_names"] = class_names
        class_info["num_classes"] = len(class_names)
    if heads_class_names is not None:
        class_info["heads_class_names"] = heads_class_names
        # Optional: derive num_classes per head
        class_info["heads_num_classes"] = {
            head: len(names) for head, names in heads_class_names.items()
        }

    # Generic metadata
    metadata: Dict[str, Any] = {
        "arch": arch_info,
        "class_info": class_info,
        "img_size": img_size,
    }
    if extra_metadata:
        metadata["extra"] = extra_metadata

    # Training state
    train_state: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }

    if optimizer is not None:
        train_state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        train_state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        train_state["scaler"] = scaler.state_dict()

    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
        "train_state": train_state,
        "format_version": 1,
    }

    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to: {save_path.resolve()}")


# ============================================================
# 2. LOAD CHECKPOINT
# ============================================================

def _build_model_from_arch_info(
    arch_info: Dict[str, Any],
    device: str | torch.device,
    override_model_builder: Optional[Callable[..., nn.Module]] = None,
) -> nn.Module:
    """
    Rebuild a model from arch_info (module, class_name, kwargs).

    If override_model_builder is provided, it's used as:
        model = override_model_builder(arch_info)

    Else:
        - dynamically imports module and gets class by name
        - instantiates with arch_info["kwargs"]
    """
    if override_model_builder is not None:
        model = override_model_builder(arch_info)
        return model.to(device)

    module_name = arch_info["module"]
    class_name = arch_info["class_name"]
    kwargs = arch_info.get("kwargs", {})

    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    model = model_cls(**kwargs)
    return model.to(device)


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    override_model_builder: Optional[Callable[[Dict[str, Any]], nn.Module]] = None,
    strict: bool = True,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Load a rich checkpoint and reconstruct the model.

    Args:
        checkpoint_path: path to .pth checkpoint
        device: device to map tensors to ("cpu", "cuda", "mps", or torch.device)
        override_model_builder:
            Optional callable taking arch_info and returning a model instance.
            Use this if:
              - you want to map a generic arch_info to a specific torchvision function
              - you changed your code but still want to load old checkpoints
        strict: passed to model.load_state_dict(strict=strict)
        optimizer: if provided, its state will be loaded from the checkpoint (if available)
        scheduler: if provided, its state will be loaded
        scaler: if provided, its state will be loaded

    Returns:
        model: nn.Module with loaded weights
        metadata: dict with "arch", "class_info", "img_size", "extra"
        train_state: dict with "epoch", "global_step", "best_metric", and possibly others
    """
    checkpoint_path = _to_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["state_dict"]
    metadata = checkpoint.get("metadata", {})
    train_state = checkpoint.get("train_state", {})

    arch_info = metadata.get("arch")
    if arch_info is None:
        raise ValueError("Checkpoint missing 'arch' info; cannot auto-rebuild model.")

    # Build model from arch info
    model = _build_model_from_arch_info(
        arch_info=arch_info,
        device=device,
        override_model_builder=override_model_builder,
    )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"✅ Model loaded from: {checkpoint_path.resolve()}")
    print(f"   Missing keys   : {missing}")
    print(f"   Unexpected keys: {unexpected}")

    # Optionally restore optimizer/scheduler/scaler
    if optimizer is not None and "optimizer" in train_state:
        optimizer.load_state_dict(train_state["optimizer"])
        print("   ✅ Optimizer state restored.")

    if scheduler is not None and "scheduler" in train_state:
        scheduler.load_state_dict(train_state["scheduler"])
        print("   ✅ Scheduler state restored.")

    if scaler is not None and "scaler" in train_state:
        scaler.load_state_dict(train_state["scaler"])
        print("   ✅ GradScaler state restored.")

    model.eval()
    return model, metadata, train_state
