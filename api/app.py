import sys
import io
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.train_model        import build_model, NUM_CLASSES
from utils.preprocessing      import preprocess_for_inference
from explainability.lime_explainer import generate_lime_explanation
from explainability.shap_explainer import generate_shap_explanation
MODEL_WEIGHTS  = Path("model/model_weights.pth")
OUTPUTS_DIR    = Path("outputs")
CLASS_NAMES    = ["Authentic", "Counterfeit"]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACKAGING_FEATURES = [
    "logo placement",
    "text alignment / typography",
    "color consistency",
    "barcode / QR presence",
    "packaging seal quality",
]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
app = FastAPI(
    title="Counterfeit Medicine Detection API",
    description="Deep-learning + LIME/SHAP explainability API for medicine package authentication.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_model: Optional[torch.nn.Module] = None
def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        if not MODEL_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{MODEL_WEIGHTS}'. "
                "Run `python model/train_model.py` first."
            )
        model = build_model(num_classes=NUM_CLASSES)
        state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        model.to(DEVICE)
        _model = model
        print(f"Model loaded from {MODEL_WEIGHTS} on {DEVICE}")
    return _model
@app.on_event("startup")
def startup_event():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("App started. Model will load on first request.")
@app.get("/health")
def health():
    model_ready = MODEL_WEIGHTS.exists()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model_ready,
    }
@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Medicine package image (JPEG/PNG)"),
    run_lime: bool = True,
    run_shap: bool = True,
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {ALLOWED_EXTENSIONS}",
        )
    tmp_path = Path(tempfile.mktemp(suffix=suffix))
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        model = _load_model()
        tensor = preprocess_for_inference(str(tmp_path)).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)[0]
        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item()
        prediction = CLASS_NAMES[pred_idx]
        class_probs = {
            CLASS_NAMES[0]: round(probs[0].item(), 4),
            CLASS_NAMES[1]: round(probs[1].item(), 4),
        }
        run_id    = uuid.uuid4().hex[:8]
        lime_path = str(OUTPUTS_DIR / f"lime_{run_id}.png")
        shap_path = str(OUTPUTS_DIR / f"shap_{run_id}.png")
        explanations = {}
        if run_lime:
            try:
                lime_out = generate_lime_explanation(
                    model, str(tmp_path), output_path=lime_path
                )
                explanations["lime_image_path"] = lime_out
            except Exception as e:
                explanations["lime_error"] = str(e)
        if run_shap:
            try:
                shap_out = generate_shap_explanation(
                    model, str(tmp_path), output_path=shap_path
                )
                explanations["shap_image_path"] = shap_out
            except Exception as e:
                explanations["shap_error"] = str(e)
        return JSONResponse(content={
            "prediction":          prediction,
            "confidence_score":    round(confidence, 4),
            "class_probabilities": class_probs,
            "features_considered": PACKAGING_FEATURES,
            "explanations":        explanations,
        })
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
@app.get("/outputs/{filename}")
def serve_output(filename: str):
    """Serve a saved explanation PNG by filename."""
    path = OUTPUTS_DIR / filename
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="image/png")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port)
