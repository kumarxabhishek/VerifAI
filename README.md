# VerifAI — AI Face Detection at Scale

> Detect whether a portrait image is real or AI-generated, with explainable verdicts.

**[Live Demo](https://verif-ai-front-end.vercel.app)** | **[Backend API](https://abhishekkumar11-verifai.hf.space)** | License: MIT

---

## What is VerifAI?

VerifAI is a binary image classifier that detects whether a portrait face image is a real photograph or AI-generated, wrapped in a lightweight multi-agent system that explains its reasoning.

Inspired by Zepto's [ZepIris](https://github.com/zepto-labs/zepiris) architecture — which uses MobileNetV3 for liveness detection — VerifAI applies the same family of models to a different problem: detecting AI-generated faces in an era where synthetic media is increasingly indistinguishable from real.

---

## Live Demo

**Try it here:** [verif-ai-front-end.vercel.app](https://verif-ai-front-end.vercel.app)

Upload any portrait face image or click an example to analyze it.

> **Note:** Model is optimized for portrait face images similar to its training distribution (StyleGAN, early Stable Diffusion outputs). Works best on close-up face portraits, not general AI art or illustrations.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 99.68% |
| F1 Score | 0.9968 |
| Precision | 99.70% |
| Recall | 99.66% |
| Test Set Size | 10,000 images |
| Best Threshold | 0.25 |

### Robustness Testing

Model evaluated under real-world image degradations:

| Degradation | F1 Score | Drop vs Clean |
|-------------|----------|---------------|
| Clean (baseline) | 0.9990 | — |
| JPEG Quality 75 | 0.9586 | -4.0% |
| JPEG Quality 50 | 0.8850 | -11.4% |
| JPEG Quality 25 | 0.7204 | -27.9% |
| JPEG Quality 10 | 0.1150 | -88.4% |
| Resize 128px | 0.5653 | -43.4% |
| Resize 64px | 0.1023 | -89.7% |
| Resize 32px | 0.0000 | -99.9% |

**Key finding:** The model relies heavily on high-frequency pixel-level features. Heavy JPEG compression and aggressive resizing destroy exactly these features — consistent with published literature on deepfake detector brittleness. Counterintuitively, the higher-accuracy 100k model showed worse robustness than the smaller 5k model, demonstrating that clean benchmark accuracy doesn't guarantee real-world reliability.

---

## Architecture

```
User uploads image
        ↓
Input Agent
- Checks resolution (minimum 64x64)
- Checks blur level (Laplacian variance)
- Routes: pass → orchestrator | fail → rejection message
        ↓
Orchestrator (parallel execution)
  ├── Detection Agent
  │   └── MobileNetV3-Large → sigmoid → confidence %
  └── Explanation Agent
      └── Texture / Edge / Color heuristics
        ↓
Final Agent
- Combines verdict + confidence + quality warning
- Produces human-readable explanation with % contributions
        ↓
React UI displays results
```

### Two Microservices

**Backend (FastAPI — HuggingFace Spaces)**
- `POST /analyze` — accepts image, runs LangGraph pipeline, returns JSON
- `GET /health` — health check

**Frontend (React — Vercel)**
- Drag and drop image upload
- Animated confidence bar
- Explanation breakdown (texture, edge, color)
- Example gallery with labeled real and AI images

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model | MobileNetV3-Large (PyTorch) |
| Training Framework | PyTorch |
| Agent Pipeline | LangGraph |
| Backend API | FastAPI |
| Frontend | React + Tailwind (Lovable) |
| Backend Hosting | Hugging Face Spaces |
| Frontend Hosting | Vercel |

---

## Model Details

**Architecture:** MobileNetV3-Large with custom binary classification head

```
MobileNetV3 backbone (pretrained ImageNet)
        ↓
AdaptiveAvgPool → 960 features
        ↓
Linear(960 → 1280) → Hardswish → Dropout(0.2)
        ↓
Linear(1280 → 1) → Sigmoid → probability
```

**Why MobileNetV3:**
Deliberately chosen for its speed-vs-capacity tradeoff. A production deepfake detector processing millions of images daily needs fast inference — not just peak accuracy. This mirrors ZepIris's architecture decision for the same reason.

**Training details:**
- Dataset: 140k Real and Fake Faces (Kaggle)
- Train: 100,000 images (50k real + 50k fake)
- Validation: 20,000 images
- Test: 20,000 images (evaluation on 10k subset)
- Optimizer: Adam (lr=0.0001)
- Loss: BCEWithLogitsLoss
- Epochs: 5
- Early layers frozen (first 5 feature layers)
- Best model saved at lowest validation loss

---

## Multi-Agent Pipeline (LangGraph)

Three agents with clear separation of concerns:

**Input Agent**
Validates image quality before running expensive ML inference. Checks resolution and blur — if either fails, rejects with explanation instead of wasting compute on an unreliable prediction.

**Detection Agent**
Runs MobileNetV3 forward pass, applies sigmoid, returns `is_ai` (bool) and `confidence` (float). Single responsibility — no explanation logic here.

**Explanation Agent**
Computes three independent heuristics:
- **Texture inconsistency** — variance of local patch variances. AI images often have unnaturally uniform texture.
- **Edge sharpness anomaly** — center vs border sharpness comparison. AI images frequently show inconsistent sharpness across regions.
- **Color distribution anomaly** — standard deviation imbalance across RGB channels. AI images cluster color distributions differently than real photographs.

Scores are normalized to percentages representing relative contribution — not absolute certainty scores.

**Final Agent**
Aggregates all results. If image quality warning exists alongside a verdict, notes that results may be unreliable — avoids giving false confidence on degraded inputs.

---

## Honest Limitations

**Distribution:** Trained on StyleGAN and early Stable Diffusion outputs. These generators leave specific pixel-level artifacts the model learned to detect. Frontier 2026 generators (Flux, Midjourney v7) produce fundamentally different outputs — current research shows even SOTA detectors achieve only 18-30% accuracy against them.

**Compression sensitivity:** Heavy JPEG compression (Q≤25) and aggressive resizing (≤64px) significantly degrade performance. Model relies on high-frequency features destroyed by lossy compression.

**Portrait faces only:** Designed for close-up face portraits, not general AI art, illustrations, or crowd scenes.

**Threshold:** Best F1 achieved at threshold=0.25 (not default 0.5), suggesting the model is overconfident in its probability outputs — a known artifact of training on large balanced datasets.

---

## Project Structure

```
verifai/
├── src/
│   ├── dataset.py          # PyTorch Dataset + DataLoader
│   ├── model.py            # MobileNetV3 classifier
│   ├── train.py            # Training loop
│   ├── evaluate.py         # F1, confusion matrix, threshold search
│   ├── robustness.py       # JPEG + resize degradation testing
│   ├── agents.py           # LangGraph multi-agent pipeline
│   └── api.py              # FastAPI backend
├── models/
│   └── best_model.pth      # Trained weights
├── Dockerfile              # HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/kumarxabhishek/VerifAI
cd VerifAI
pip install -r requirements.txt
```

Download dataset from [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) and place in `data/real-vs-fake/`.

```bash
# Train
python src/train.py

# Evaluate
python src/evaluate.py

# Robustness test
python src/robustness.py

# Run API locally
python src/api.py
```

---

## References

- [MobileNetV3 — Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [FaceForensics++ — Learning to Detect Manipulated Facial Images](https://arxiv.org/abs/1901.08971)
- [DeepFakes and Beyond — A Survey](https://arxiv.org/abs/2001.00179)
- [Unmasking DeepFakes with Simple Features](https://arxiv.org/abs/1911.00686)
- [ZepIris — Zepto's Face Authentication Platform](https://github.com/zepto-labs/zepiris)

---

## Built By

**Abhishek Kumar**  
[LinkedIn](https://www.linkedin.com/in/abhishek-kumar-42a2a024a/)
