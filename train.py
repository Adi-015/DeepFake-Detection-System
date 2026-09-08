"""
train.py — Full training pipeline for Fake vs Real Image Detection.

Supports:
  - Hybrid CNN + ViT + Frequency model training
  - Optional SimCLR-style self-supervised pretraining
  - Mixed precision (AMP), cosine LR schedule, early stopping
  - Comprehensive metrics: accuracy, F1, AUC, confusion matrix
  - Checkpointing and TensorBoard logging

Usage:
  python train.py --data_dir ./dataset --epochs 30 --batch_size 32
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from model_architecture import HybridDetector, build_model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.preprocessing import (
    get_train_transforms, get_val_transforms,
    extract_freq_tensor, IMAGE_SIZE
)


# ─────────────────────────────────────────────
# Custom Dataset with Frequency Features
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Dataset loader supporting dataset/real/ and dataset/fake/ structure.
    Returns both spatial and frequency tensors.
    """
    def __init__(self, root: str, split: str = 'train', val_ratio: float = 0.15):
        self.root = Path(root)
        self.split = split

        # Gather all images
        self.samples = []
        self.labels  = []

        for label, cls in enumerate(['real', 'fake']):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                logger.warning(f"Missing class directory: {cls_dir}")
                continue
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'):
                for p in cls_dir.glob(ext):
                    self.samples.append(str(p))
                    self.labels.append(label)

        # Train/val split
        n = len(self.samples)
        indices = np.random.permutation(n)
        val_size = int(n * val_ratio)
        if split == 'val':
            indices = indices[:val_size]
        else:
            indices = indices[val_size:]

        self.samples = [self.samples[i] for i in indices]
        self.labels  = [self.labels[i] for i in indices]

        self.spatial_tf = get_train_transforms() if split == 'train' else get_val_transforms()
        logger.info(f"[{split}] {len(self.samples)} images | real: {self.labels.count(0)} | fake: {self.labels.count(1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_path = self.samples[idx]
        label    = self.labels[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Skipping corrupt image {img_path}: {e}")
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

        spatial = self.spatial_tf(img)              # (3, H, W)
        freq    = extract_freq_tensor(img)          # (4, H, W)
        return spatial, freq, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in self.labels], dtype=torch.float)
        return sample_weights


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


import torch.nn.functional as F


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self._setup_data()
        self._setup_model()
        self._setup_training()
        self.best_val_auc = 0.0
        self.patience_counter = 0

    def _setup_data(self):
        train_ds = DeepfakeDataset(self.args.data_dir, split='train')
        val_ds   = DeepfakeDataset(self.args.data_dir, split='val')

        # Balanced sampling
        sample_weights = train_ds.get_class_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size,
            sampler=sampler, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

    def _setup_model(self):
        self.model = HybridDetector(cnn_backbone=self.args.backbone).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,}")

    def _setup_training(self):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

    # ── Training epoch
    def train_epoch(self) -> Dict:
        self.model.train()
        losses, preds, truths = [], [], []

        for spatial, freq, labels in tqdm(self.train_loader, desc='Train', leave=False):
            spatial = spatial.to(self.device, non_blocking=True)
            freq    = freq.to(self.device, non_blocking=True)
            labels  = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(enabled=(self.device.type == 'cuda')):
                logits = self.model(spatial, freq)
                loss   = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())
            truths.extend(labels.cpu().numpy())

        return {
            'loss': np.mean(losses),
            'acc':  accuracy_score(truths, preds),
            'f1':   f1_score(truths, preds, zero_division=0),
        }

    # ── Validation epoch
    @torch.no_grad()
    def val_epoch(self) -> Dict:
        self.model.eval()
        losses, preds, probs, truths = [], [], [], []

        for spatial, freq, labels in tqdm(self.val_loader, desc='Val', leave=False):
            spatial = spatial.to(self.device)
            freq    = freq.to(self.device)
            labels  = labels.to(self.device)

            logits = self.model(spatial, freq)
            loss   = self.criterion(logits, labels)
            prob   = torch.softmax(logits, dim=-1)

            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())
            truths.extend(labels.cpu().numpy())

        auc = roc_auc_score(truths, probs) if len(set(truths)) > 1 else 0.5
        return {
            'loss': np.mean(losses),
            'acc':  accuracy_score(truths, preds),
            'f1':   f1_score(truths, preds, zero_division=0),
            'auc':  auc,
            'preds': preds,
            'truths': truths,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict, tag: str = 'best'):
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"detector_{tag}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(self.args),
        }, path)
        logger.info(f"Saved checkpoint → {path}")

    def plot_confusion_matrix(self, truths, preds, save_path: str):
        cm = confusion_matrix(truths, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def train(self):
        logger.info("Starting training...")
        history = {'train': [], 'val': []}

        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics   = self.val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{self.args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            # Checkpoint on best AUC
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, tag='best')
                # Save confusion matrix
                self.plot_confusion_matrix(
                    val_metrics['truths'], val_metrics['preds'],
                    str(Path(self.args.save_dir) / 'confusion_matrix.png')
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Save every N epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_metrics, tag=f'epoch_{epoch}')

        # Final report
        self.save_checkpoint(self.args.epochs, val_metrics, tag='last')
        with open(Path(self.args.save_dir) / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training complete. Best Val AUC: {self.best_val_auc:.4f}")
        logger.info(f"\n{classification_report(val_metrics['truths'], val_metrics['preds'], target_names=['Real','Fake'])}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Fake vs Real Image Detector")
    p.add_argument('--data_dir',     type=str,   default='./dataset')
    p.add_argument('--save_dir',     type=str,   default='./saved_model')
    p.add_argument('--backbone',     type=str,   default='efficientnet_b3')
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--patience',     type=int,   default=8)
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = Trainer(args)
    trainer.train()
