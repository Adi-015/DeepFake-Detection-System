# 🔍 Fake vs Real Image Detector

A **research-grade backend** for detecting AI-generated images using a **Hybrid CNN + Vision Transformer** with **frequency-domain (FFT/DCT)** analysis, GradCAM explainability, and a production-ready FastAPI server.

---

## Architecture Overview

```
Input Image
    │
    ├─── [Stream A] CNN Backbone (EfficientNet-B3)
    │         └─► Spatial features (512-d)
    │
    ├─── [Stream B] Vision Transformer
    │         ├─► Patch Embedding (16×16 patches)
    │         ├─► 6× Transformer Blocks (Multi-Head Attention)
    │         └─► CLS Token features (512-d)
    │
    └─── [Stream C] Frequency Branch
              ├─► FFT magnitude spectrum
              ├─► DCT channel features
              └─► Frequency features (256-d)
                       │
              ┌────────▼────────┐
              │  Fusion MLP     │  (1280-d → 512 → 256 → 2)
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Prediction     │  Real / Fake + confidence
              └─────────────────┘
```

### Why this architecture?
- **CNN backbone** captures local texture and artifact patterns that AI models leave behind
- **ViT transformer** captures global consistency — AI images often have subtle long-range inconsistencies
- **Frequency branch** detects grid artifacts, frequency peaks, and spectral anomalies common in GAN/Diffusion outputs

---

## Project Structure

```
backend/
├── app.py                    # FastAPI application
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── model/
│   ├── model_architecture.py # HybridDetector (CNN + ViT + Freq)
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Inference + GradCAM
│   └── model_loader.py       # Singleton model cache
├── utils/
│   └── preprocessing.py      # FFT/DCT, augmentation, transforms
├── saved_model/              # Trained checkpoints
├── heatmaps/                 # GradCAM outputs
└── logs/                     # Application logs
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset
```
dataset/
├── real/
│   ├── img001.jpg
│   └── ...
└── fake/
    ├── img001.jpg
    └── ...
```

### 3. Train the model
```bash
cd model
python train.py \
  --data_dir ./dataset \
  --save_dir ../saved_model \
  --backbone efficientnet_b3 \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4
```

### 4. Start the API
```bash
# Dev
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production
python app.py
```

### 5. Docker
```bash
docker-compose up --build
```

---

## API Usage

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_image.jpg" \
  -F "generate_heatmap=true"
```

### Response
```json
{
  "prediction": "Fake",
  "confidence": 0.92,
  "probabilities": {
    "Real": 0.08,
    "Fake": 0.92
  },
  "artifact_score": 0.78,
  "heatmap_url": "http://localhost:8000/heatmaps/heatmap_3a8f12bc.jpg",
  "processing_time": 0.45,
  "image_info": {
    "width": 1024,
    "height": 768,
    "format": "JPEG"
  }
}
```

### GET /health
```bash
curl http://localhost:8000/health
```

Interactive docs available at: `http://localhost:8000/docs`

---

## Training Details

### Loss Function
- **Cross-Entropy with label smoothing (0.1)** — reduces overconfidence
- Optionally switch to **Focal Loss** for class imbalance

### Optimizer & Schedule
- **AdamW** (weight_decay=1e-2)
- **Cosine Annealing with Warm Restarts** (T_0=10)
- Gradient clipping (max norm=1.0)

### Regularization
- Weighted random sampling for class balance
- Label smoothing
- Dropout (0.2 throughout)
- Mixed precision (AMP)

### Augmentation
- Random horizontal/vertical flip
- Color jitter, random grayscale
- Random rotation, Gaussian blur

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Accuracy** | Overall correct predictions |
| **F1 Score** | Balance of precision/recall |
| **AUC-ROC** | Discriminative ability |
| **Confusion Matrix** | Error pattern breakdown |

Run after training:
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
```

---

## Frequency Analysis

AI-generated images exhibit characteristic patterns in frequency domain:

- **GAN images**: Show checkerboard artifacts at specific frequencies (grid artifacts from upsampling)
- **Diffusion models**: Show smoother but anomalous spectral distributions
- **Face synthesis**: Exhibit eye/teeth region spectral inconsistencies

The **FFT branch** detects these via magnitude spectrum analysis. The **artifact_score** in the response reflects the ratio of high-frequency energy to total energy — synthetic images tend to score higher.

---

## Explainable AI (XAI)

**GradCAM** highlights image regions that most influenced the prediction:
- 🔴 Red/hot regions → highest contribution
- 🔵 Blue/cool regions → low contribution

Enable with `generate_heatmap=true` in the API call.

---

## Research Enhancements

### Ensemble Options
```python
# Combine multiple models at inference:
# 1. HybridDetector (CNN+ViT+Freq) — this implementation
# 2. Pure ViT (ViT-B/16 fine-tuned)
# 3. ResNet50 + SRM (Steganalysis Rich Model) filter
# Average softmax probabilities for final prediction
```

### Self-Supervised Pretraining (SimCLR)
```python
# The ContrastiveHead in model_architecture.py enables SimCLR pretraining:
# 1. Pretrain on unlabeled images using contrastive loss
# 2. Fine-tune on labeled real/fake dataset
# This is especially useful when labeled data is scarce
```

### Recommended Datasets

| Dataset | Description | Size |
|---|---|---|
| **CIFAKE** | CIFAR-10 real vs Stable Diffusion | 120K |
| **FaceForensics++** | Video deepfakes (face manipulation) | 1.8M frames |
| **DALL-E 3 / Midjourney** | Prompt-generated images | Community |
| **GenImage** | 8 generators × 1M images | 1M+ |
| **WildDeepfake** | In-the-wild face videos | 707 videos |
| **DFFD** | Diverse face forgery dataset | 299K |

---

## Deployment Recommendations

### Cloud (AWS)
```
EC2 g4dn.xlarge (T4 GPU) → ~$0.53/hr
Load Balancer + Auto Scaling for production traffic
S3 bucket for heatmap storage (replace local files)
```

### Performance
- CPU inference: ~0.4–0.8s/image
- GPU inference (T4): ~0.05–0.15s/image
- Batch inference: use DataLoader pipeline in train.py

### Scaling
```bash
# Multi-worker Gunicorn + Uvicorn
gunicorn app:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `saved_model/detector_best.pt` | Path to checkpoint |
| `BASE_URL` | `http://localhost:8000` | Base URL for heatmap links |
| `MAX_FILE_SIZE_MB` | `20` | Max upload size |
