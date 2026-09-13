"""
preprocessing.py — Image preprocessing, augmentation, and frequency-domain feature extraction.
Handles spatial + FFT/DCT features for AI-generated image detection.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
import cv2
import io
import time
from typing import Tuple, Dict, Any, Optional
from loguru import logger
from scipy.fft import fft2, fftshift
from scipy.fftpack import dct


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
PATCH_SIZE = 16
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Base Transforms
# ─────────────────────────────────────────────
def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.RandomRotation(degrees=10),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


# ─────────────────────────────────────────────
# Frequency Domain Features
# ─────────────────────────────────────────────
def compute_fft_features(img_np: np.ndarray) -> np.ndarray:
    """Compute 2D FFT magnitude spectrum. Shape: (1, H, W)"""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
    gray = gray.astype(np.float32) / 255.0
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    # Normalize
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude[np.newaxis, :, :]  # (1, H, W)


def compute_dct_features(img_np: np.ndarray) -> np.ndarray:
    """Compute 2D DCT features channel-wise. Shape: (3, H, W)"""
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    result = np.zeros_like(img_np, dtype=np.float32)
    for c in range(3):
        channel = img_np[:, :, c].astype(np.float32)
        d = dct(dct(channel, axis=0, norm='ortho'), axis=1, norm='ortho')
        d = np.log1p(np.abs(d))
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        result[:, :, c] = d
    return result.transpose(2, 0, 1)  # (3, H, W)


def extract_freq_tensor(pil_img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Extract combined FFT + DCT frequency features as a 4-channel tensor."""
    img = pil_img.convert("RGB").resize((size, size))
    img_np = np.array(img)

    fft_feat = compute_fft_features(img_np)   # (1, H, W)
    dct_feat = compute_dct_features(img_np)   # (3, H, W)
    combined = np.concatenate([fft_feat, dct_feat], axis=0)  # (4, H, W)
    return torch.tensor(combined, dtype=torch.float32)


# ─────────────────────────────────────────────
# Patch Extraction (ViT-style)
# ─────────────────────────────────────────────
def extract_patches(tensor: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Extract non-overlapping patches from image tensor.
    Input:  (C, H, W)
    Output: (N, C, P, P)  where N = (H/P)*(W/P)
    """
    C, H, W = tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # (N, C, P, P)
    return patches


# ─────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────
def preprocess_for_inference(
    image_bytes: bytes,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Full preprocessing pipeline for inference.
    Returns:
        spatial_tensor  : (1, 3, H, W)
        freq_tensor     : (1, 4, H, W)
        metadata        : dict with image info
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"Invalid image: {e}")

    metadata = {
        "width":  pil_img.width,
        "height": pil_img.height,
        "format": pil_img.format or "UNKNOWN",
        "mode":   pil_img.mode,
    }

    # Spatial tensor
    transforms = get_val_transforms()
    spatial = transforms(pil_img).unsqueeze(0)   # (1, 3, H, W)

    # Frequency tensor
    freq = extract_freq_tensor(pil_img).unsqueeze(0)  # (1, 4, H, W)

    return spatial, freq, metadata


def compute_artifact_score(freq_tensor: torch.Tensor) -> float:
    """
    Heuristic artifact score based on high-frequency energy ratio.
    AI-generated images often exhibit artifacts in specific frequency bands.
    """
    freq_np = freq_tensor.squeeze(0).numpy()  # (4, H, W)
    fft_channel = freq_np[0]
    H, W = fft_channel.shape
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 8

    # Mask high-frequency region (away from center)
    y, x = np.ogrid[:H, :W]
    low_mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    high_mask = ~low_mask

    total_energy = fft_channel.sum() + 1e-8
    high_energy = fft_channel[high_mask].sum()
    score = float(high_energy / total_energy)
    return min(max(score, 0.0), 1.0)


def validate_image(image_bytes: bytes) -> bool:
    """Basic validation that bytes represent a readable image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False
