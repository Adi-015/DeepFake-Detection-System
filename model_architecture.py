"""
model_architecture.py — Hybrid CNN + Vision Transformer for Fake vs Real Image Detection.

Architecture:
  1. CNN Backbone (EfficientNet/ConvNeXt) → spatial feature extraction
  2. Vision Transformer encoder → global context + patch attention
  3. Frequency Branch (FFT/DCT) → artifact pattern detection
  4. Fusion head → combined prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from typing import Optional


# ─────────────────────────────────────────────
# Multi-Head Self-Attention Block
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ─────────────────────────────────────────────
# Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Frequency Analysis Branch
# ─────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    """Processes FFT + DCT 4-channel frequency input."""
    def __init__(self, in_channels: int = 4, out_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))


# ─────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, dim: int = 768):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_size * patch_size * in_channels),
            nn.Linear(patch_size * patch_size * in_channels, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.dropout   = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return self.dropout(x)


# ─────────────────────────────────────────────
# CNN Backbone Wrapper
# ─────────────────────────────────────────────
class CNNBackbone(nn.Module):
    """Wraps a pretrained EfficientNet/ConvNeXt for feature extraction."""
    def __init__(self, backbone_name: str = 'efficientnet_b3', out_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool='avg'
        )
        backbone_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


# ─────────────────────────────────────────────
# Main Hybrid Model
# ─────────────────────────────────────────────
class HybridDetector(nn.Module):
    """
    Hybrid CNN + Transformer + Frequency Branch for deepfake detection.

    Streams:
      A) CNN backbone → spatial features (512-d)
      B) ViT patches → global attention features (768-d)
      C) Frequency branch → FFT/DCT features (256-d)

    Fusion: concatenate → MLP head → binary output
    """
    def __init__(
        self,
        image_size:    int = 224,
        patch_size:    int = 16,
        cnn_backbone:  str = 'efficientnet_b3',
        vit_dim:       int = 512,
        vit_heads:     int = 8,
        vit_depth:     int = 6,
        cnn_out_dim:   int = 512,
        freq_out_dim:  int = 256,
        dropout:       float = 0.2,
        num_classes:   int = 2,
    ):
        super().__init__()

        # ── Stream A: CNN backbone
        self.cnn = CNNBackbone(cnn_backbone, cnn_out_dim)

        # ── Stream B: ViT
        self.patch_embed    = PatchEmbedding(image_size, patch_size, in_channels=3, dim=vit_dim)
        self.transformer    = nn.Sequential(
            *[TransformerBlock(vit_dim, vit_heads, dropout=dropout) for _ in range(vit_depth)]
        )
        self.vit_norm       = nn.LayerNorm(vit_dim)

        # ── Stream C: Frequency
        self.freq_branch    = FrequencyBranch(in_channels=4, out_dim=freq_out_dim)

        # ── Fusion
        fusion_dim = cnn_out_dim + vit_dim + freq_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        spatial:   torch.Tensor,          # (B, 3, H, W)
        freq:      torch.Tensor,          # (B, 4, H, W)
        return_attn: bool = False,
    ):
        # CNN stream
        cnn_feat = self.cnn(spatial)                           # (B, 512)

        # ViT stream
        tokens  = self.patch_embed(spatial)                    # (B, N+1, D)
        tokens  = self.transformer(tokens)
        tokens  = self.vit_norm(tokens)
        vit_feat = tokens[:, 0]                                # CLS token (B, D)

        # Frequency stream
        freq_feat = self.freq_branch(freq)                     # (B, 256)

        # Fusion
        fused = torch.cat([cnn_feat, vit_feat, freq_feat], dim=1)
        logits = self.fusion(fused)                            # (B, 2)

        if return_attn:
            return logits, tokens[:, 1:]  # return patch tokens for visualization
        return logits

    def get_attention_weights(self, spatial: torch.Tensor) -> torch.Tensor:
        """Extract attention map from last transformer block for GradCAM."""
        tokens = self.patch_embed(spatial)
        for block in self.transformer:
            tokens = block(tokens)
        return tokens[:, 1:]  # patch tokens only


# ─────────────────────────────────────────────
# Contrastive Pretraining Head (SimCLR-style)
# ─────────────────────────────────────────────
class ContrastiveHead(nn.Module):
    """
    Projection head for self-supervised pretraining.
    Attach to HybridDetector.cnn for SimCLR/MoCo-style pretraining.
    """
    def __init__(self, in_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(x), dim=-1)


def build_model(
    backbone: str = 'efficientnet_b3',
    pretrained_path: Optional[str] = None,
    device: str = 'cpu',
) -> HybridDetector:
    model = HybridDetector(cnn_backbone=backbone)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=False)
    return model.to(device)
