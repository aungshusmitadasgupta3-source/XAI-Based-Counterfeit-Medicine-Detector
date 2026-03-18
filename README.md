# Counterfeit Medicine Detection System

A production-grade pipeline that uses a fine-tuned **EfficientNet-B0** CNN to
classify medicine packaging images as **Authentic** or **Counterfeit**, with
**LIME** and **SHAP** explainability visualisations.

---

## Project Structure

```
project/
│
├── model/
│   ├── train_model.py          ← Training pipeline (EfficientNet-B0)
│   └── model_weights.pth       ← Saved weights (generated after training)
│
├── explainability/
│   ├── lime_explainer.py       ← LIME explanation generator
│   └── shap_explainer.py       ← SHAP explanation generator
│
├── api/
│   └── app.py                  ← FastAPI REST endpoint
│
├── utils/
│   ├── preprocessing.py        ← Image transforms & helpers
│   └── generate_demo_data.py   ← Synthetic dataset generator (for testing)
│
├── outputs/
│   ├── lime_explanation.png    ← Generated at runtime
│   └── shap_explanation.png    ← Generated at runtime
│
├── predict.py                  ← Standalone CLI (no server needed)
└── requirements.txt
```

---

## Quick Start - Step-by-Step

### Step 0 - Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 1 - Prepare your dataset

Arrange them like this:
```
data/
  train/
    authentic/    ← real medicine package photos
    counterfeit/  ← fake/counterfeit medicine package photos
  val/
    authentic/
    counterfeit/
``` vb

### Step 2 — Train the model

```bash
python model/train_model.py \
    --data_dir   data \
    --save_path  model/model_weights.pth \
    --epochs     20 \
    --batch_size 32 \
    --lr         0.001
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data` | Root folder with train/val splits |
| `--save_path` | `model/model_weights.pth` | Where to save the best weights |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `0.001` | Initial learning rate |
| `--freeze` | off | Freeze backbone (pure feature-extraction mode) |

The best checkpoint (highest val accuracy) is saved automatically.
A `training_history.json` with loss/accuracy curves is also written.

---

### Step 3A - Run predictions 

```bash
python predict.py --image path/to/medicine.jpg
```

Optional flags:
```bash
python predict.py --image medicine.jpg --skip_lime   # skip LIME
python predict.py --image medicine.jpg --skip_shap   # skip SHAP
```

Output is printed to console **and** saved to `outputs/result.json`.
Explanation PNGs are written to `outputs/`.

---

### Step 3B - Start the REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then visit **http://localhost:8000/docs** for the interactive Swagger UI.

**POST /predict** - upload an image:
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@medicine.jpg" \
     -F "run_lime=true" \
     -F "run_shap=true"
```

**GET /outputs/{filename}** — retrieve a saved explanation PNG:
```
http://localhost:8000/outputs/lime_explanation.png
```

---

##  Sample JSON Response

```json
{
  "prediction": "Counterfeit",
  "confidence_score": 0.9341,
  "class_probabilities": {
    "Authentic": 0.0659,
    "Counterfeit": 0.9341
  },
  "features_considered": [
    "logo placement",
    "text alignment / typography",
    "color consistency",
    "barcode / QR presence",
    "packaging seal quality"
  ],
  "explanations": {
    "lime_image_path": "outputs/lime_explanation.png",
    "shap_image_path": "outputs/shap_explanation.png"
  }
}
```

---

##  How the Explainability Works

### LIME — Local Interpretable Model-Agnostic Explanations

1. Takes the input image and divides it into **superpixels** (compact regions of similar pixels, using Quickshift segmentation).
2. Creates ~1000 **perturbations** by randomly masking different subsets of superpixels to grey.
3. Runs every perturbation through the CNN → collects predicted probabilities.
4. Fits a **lightweight linear model** on (perturbation, prediction) pairs, weighting by proximity to the original.
5. The linear model's coefficients reveal which superpixels most strongly pushed the prediction toward "Authentic" or "Counterfeit".

**Output**: Three-panel PNG —
- Original image
- Highlighted superpixels supporting the predicted class
- Superpixel weight heatmap (green = supports, red = opposes)

**Medicine application**: LIME will light up regions like the logo block, the barcode strip, or the text area — exactly the zones a pharmacist would inspect for fakes.

---

### SHAP — SHapley Additive exPlanations

1. Based on **Shapley values** from cooperative game theory: *"How much does each pixel contribute to the prediction, averaged over all possible coalitions of other pixels?"*
2. Uses **GradientExplainer** — back-propagates through the model to efficiently approximate Shapley values without enumerating all 2^N subsets.
3. A **background baseline** (near-black random noise images) represents the *absence* of a feature.
4. For each pixel: positive SHAP = pushed toward predicted class; negative SHAP = pushed away.

**Output**: Six-panel PNG —
- Original image
- SHAP value heatmap (coolwarm)
- Blended overlay
- Per-channel (R/G/B) absolute contribution maps

**Medicine application**: SHAP reveals *exact pixel-level* contributions — it will highlight a washed-out barcode, wrong brand colours, or a missing tamper-evident seal as the key evidence for a "Counterfeit" verdict.

---

### Why XAI Matters for Counterfeit Detection

| Without XAI | With XAI |
|-------------|----------|
| "Counterfeit — 94% confidence" | "Counterfeit — 94%: barcode distortion (+0.41 SHAP), off-brand colour block (+0.29), missing seal (-0.18 vs authentic)" |
| Regulatory submission fails | Explainable audit trail |
| Pharmacist can't trust or verify | Pharmacist can visually confirm |
| Model bias undetectable | Bias immediately visible |

---

##  Model Architecture

```
EfficientNet-B0 (ImageNet pretrained)
    ↓
features.*      (backbone — fine-tuned at 10× lower LR)
    ↓
AdaptiveAvgPool
    ↓
Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→2)
    ↓
Softmax → [P(Authentic), P(Counterfeit)]
```

- **Loss**: CrossEntropy with label smoothing (0.1)
- **Optimizer**: AdamW with separate LR groups (backbone 1e-4, head 1e-3)
- **Scheduler**: CosineAnnealingLR
- **Augmentation**: RandomCrop, HorizontalFlip, ColorJitter, RandomRotation

---

##  Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
lime>=0.2.0.1
shap>=0.43.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=10.0.0
scikit-learn>=1.3.0
fastapi>=0.103.0
uvicorn>=0.23.0
python-multipart>=0.0.6
tqdm>=4.66.0
```

GPU (CUDA) is used automatically if available; CPU fallback is fully supported.
