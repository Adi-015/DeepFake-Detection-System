"""
predict.py — Inference pipeline with GradCAM heatmap, confidence, artifact scoring.

Returns:
  prediction, confidence, class probabilities, artifact score,
  heatmap path, processing time, image metadata.
"""

import time
import uuid
import os
import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from loguru import logger

from model_loader import load_model, get_device
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.preprocessing import (
    preprocess_for_inference,
    compute_artifact_score,
    validate_image,
    IMAGE_SIZE,
)

CLASSES = ['Real', 'Fake']
HEATMAP_DIR = Path(__file__).parent.parent / 'heatmaps'
HEATMAP_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# GradCAM Implementation
# ─────────────────────────────────────────────
class GradCAM:
    """
    GradCAM for the CNN backbone's last convolutional layer.
    Highlights regions that influenced the prediction.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Target: last conv layer of CNN backbone
        try:
            target_layer = self.model.cnn.backbone.conv_head  # EfficientNet
        except AttributeError:
            try:
                target_layer = list(self.model.cnn.backbone.children())[-3]
            except Exception:
                target_layer = None
                logger.warning("Could not attach GradCAM hooks — heatmap unavailable.")

        if target_layer is not None:
            target_layer.register_forward_hook(self._save_activation)
            target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        spatial: torch.Tensor,
        freq: torch.Tensor,
        class_idx: int,
        original_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Returns heatmap array (H, W, 3) in BGR, or None on failure."""
        if self.activations is None:
            return None

        self.model.zero_grad()
        logits = self.model(spatial, freq)
        score  = logits[0, class_idx]
        score.backward()

        if self.gradients is None:
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to original image size
        h, w = original_size[1], original_size[0]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        return heatmap


# ─────────────────────────────────────────────
# Main Prediction Function
# ─────────────────────────────────────────────
def predict(
    image_bytes: bytes,
    model_path: Optional[str] = None,
    generate_heatmap: bool = True,
    base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """
    Full inference pipeline.

    Args:
        image_bytes      : raw image bytes
        model_path       : optional path to checkpoint
        generate_heatmap : whether to generate GradCAM
        base_url         : base URL for heatmap link

    Returns:
        dict with prediction, confidence, probabilities, artifact_score,
        heatmap_url, processing_time, image_info
    """
    t_start = time.perf_counter()

    # ── Validate
    if not validate_image(image_bytes):
        raise ValueError("Invalid or unreadable image file.")

    # ── Preprocess
    spatial, freq, metadata = preprocess_for_inference(image_bytes)
    device = get_device()
    spatial = spatial.to(device)
    freq    = freq.to(device)

    # ── Load model
    model = load_model(model_path)
    model.eval()

    # ── Inference
    with torch.no_grad():
        logits = model(spatial, freq)
        probs  = F.softmax(logits, dim=-1).squeeze(0)

    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item()
    prob_real   = probs[0].item()
    prob_fake   = probs[1].item()
    prediction  = CLASSES[pred_idx]

    # ── Artifact score (frequency-based)
    artifact_score = compute_artifact_score(freq.cpu())

    # ── GradCAM heatmap
    heatmap_url = None
    if generate_heatmap:
        try:
            model.zero_grad()
            spatial_grad = spatial.clone().requires_grad_(True)

            gradcam = GradCAM(model)
            heatmap_arr = gradcam.generate(
                spatial_grad, freq, pred_idx,
                original_size=(metadata['width'], metadata['height'])
            )

            if heatmap_arr is not None:
                # Blend with original image
                orig_img = np.array(
                    Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(
                        (metadata['width'], metadata['height'])
                    )
                )
                orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
                overlay  = cv2.addWeighted(orig_bgr, 0.6, heatmap_arr, 0.4, 0)

                fname = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
                fpath = HEATMAP_DIR / fname
                cv2.imwrite(str(fpath), overlay)
                heatmap_url = f"{base_url}/heatmaps/{fname}"
        except Exception as e:
            logger.warning(f"GradCAM failed: {e}")

    t_end = time.perf_counter()

    return {
        "prediction":      prediction,
        "confidence":      round(confidence, 4),
        "probabilities":   {"Real": round(prob_real, 4), "Fake": round(prob_fake, 4)},
        "artifact_score":  round(artifact_score, 4),
        "heatmap_url":     heatmap_url,
        "processing_time": round(t_end - t_start, 4),
        "image_info": {
            "width":  metadata['width'],
            "height": metadata['height'],
            "format": metadata['format'],
        },
    }
