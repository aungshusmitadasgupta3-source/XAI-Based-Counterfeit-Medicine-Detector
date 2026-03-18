import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from model.train_model             import build_model, NUM_CLASSES
from utils.preprocessing           import preprocess_for_inference
from explainability.lime_explainer import generate_lime_explanation
from explainability.shap_explainer import generate_shap_explanation
MODEL_WEIGHTS = Path("model/model_weights.pth")
OUTPUTS_DIR   = Path("outputs")
CLASS_NAMES   = ["Authentic", "Counterfeit"]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACKAGING_FEATURES = [
    "logo placement",
    "text alignment / typography",
    "color consistency",
    "barcode / QR presence",
    "packaging seal quality",
]
def load_model() -> torch.nn.Module:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found at '{MODEL_WEIGHTS}'. "
            "Run `python model/train_model.py` first."
        )
    model = build_model(num_classes=NUM_CLASSES)
    state = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded on {DEVICE}")
    return model
def run_pipeline(image_path: str, skip_lime: bool = False, skip_shap: bool = False) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    tensor = preprocess_for_inference(image_path).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    prediction = CLASS_NAMES[pred_idx]
    print(f"\n  Prediction : {prediction}")
    print(f"    Confidence : {confidence*100:.1f}%")
    print(f"    Authentic  : {probs[0].item()*100:.1f}%  |  Counterfeit : {probs[1].item()*100:.1f}%")
    explanations: dict = {}
    if not skip_lime:
        print("\n   Running LIME …")
        lime_path = generate_lime_explanation(
            model, image_path,
            output_path=str(OUTPUTS_DIR / "lime_explanation.png"),
        )
        explanations["lime_image_path"] = lime_path
    if not skip_shap:
        print("   Running SHAP …")
        shap_path = generate_shap_explanation(
            model, image_path,
            output_path=str(OUTPUTS_DIR / "shap_explanation.png"),
        )
        explanations["shap_image_path"] = shap_path
    result = {
        "prediction":          prediction,
        "confidence_score":    round(confidence, 4),
        "class_probabilities": {
            CLASS_NAMES[0]: round(probs[0].item(), 4),
            CLASS_NAMES[1]: round(probs[1].item(), 4),
        },
        "features_considered": PACKAGING_FEATURES,
        "explanations":        explanations,
    }
    json_path = OUTPUTS_DIR / "result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved → {json_path}")
    print(json.dumps(result, indent=2))
    return result
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfeit Medicine Detection — CLI")
    parser.add_argument("--image",     required=True,      help="Path to input image")
    parser.add_argument("--skip_lime", action="store_true", help="Skip LIME explanation")
    parser.add_argument("--skip_shap", action="store_true", help="Skip SHAP explanation")
    args = parser.parse_args()

    run_pipeline(args.image, args.skip_lime, args.skip_shap)
