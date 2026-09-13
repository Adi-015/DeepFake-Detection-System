import torch
import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Ensure backend root directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from model.model_architecture import HybridDetector


_MODEL_CACHE: Optional[HybridDetector] = None
_DEVICE: Optional[torch.device] = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _DEVICE = torch.device('mps')
            logger.info("Apple Silicon MPS detected")
        else:
            _DEVICE = torch.device('cpu')
            logger.info("Using CPU inference")
    return _DEVICE


def load_model(
    model_path: Optional[str] = None,
    backbone: str = 'efficientnet_b3',
    force_reload: bool = False,
) -> HybridDetector:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    device = get_device()
    model = HybridDetector(cnn_backbone=backbone)

    if model_path is None:
        candidates = [
            Path(__file__).parent.parent / 'saved_model' / 'detector_best.pt',
            Path(__file__).parent.parent / 'saved_model' / 'detector_last.pt',
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")
        logger.info(f"Loaded checkpoint from {model_path}")
    else:
        logger.warning("No checkpoint found — running with untrained weights (dev mode).")

    model = model.to(device)
    model.eval()

    # Warmup pass
    dummy_spatial = torch.zeros(1, 3, 224, 224).to(device)
    dummy_freq    = torch.zeros(1, 4, 224, 224).to(device)
    with torch.no_grad():
        _ = model(dummy_spatial, dummy_freq)
    logger.info("Model warmup pass completed.")

    _MODEL_CACHE = model
    return model


def unload_model():
    global _MODEL_CACHE
    _MODEL_CACHE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded from cache.")
