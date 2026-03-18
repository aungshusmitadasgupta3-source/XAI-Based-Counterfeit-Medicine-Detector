"""
explainability/shap_explainer.py
---------------------------------
SHAP (SHapley Additive exPlanations) for the
Counterfeit Medicine Detection System.

How SHAP works
--------------
SHAP is grounded in cooperative game theory (Shapley values).  For each
pixel / feature it asks:  "On average, across ALL possible coalitions of
other features, how much does INCLUDING this feature change the model output?"

For images, SHAP's GradientExplainer (or DeepExplainer) uses back-propagation
gradients to efficiently approximate these values without enumerating all
2^N feature coalitions.

Positive SHAP values (warm colours) mean the pixel/region pushed the model
toward the predicted class.  Negative values (cool colours) mean it pushed
away from it.

In medicine-packaging context, SHAP can highlight inconsistent colour blocks,
missing/distorted barcodes, or misaligned text — the exact features a domain
expert would scrutinise.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import shap
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocessing import (
    preprocess_for_inference,
    preprocess_numpy_for_lime,
    denormalise_tensor,
    IMG_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# SHAP wrapper — makes the model accept numpy
# ─────────────────────────────────────────────
class _ModelWrapper(torch.nn.Module):
    """Wraps the model so SHAP can call it with a numpy batch."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def generate_shap_explanation(
    model: torch.nn.Module,
    image_path: str,
    output_path: str = "outputs/shap_explanation.png",
    n_background: int = 20,
    n_steps: int = 50,
) -> str:
    """
    Generate a SHAP GradientExplainer explanation image.

    Parameters
    ----------
    model        : trained PyTorch model (eval mode)
    image_path   : path to the input image
    output_path  : where to save the PNG
    n_background : number of random background samples (reference baseline)
    n_steps      : gradient integration steps

    Returns
    -------
    output_path  : path of the saved PNG
    """
    model.eval()
    model.to(DEVICE)
    wrapped = _ModelWrapper(model).to(DEVICE)

    # ── 1. Prepare input tensor ───────────────────────────────────────
    input_tensor = preprocess_for_inference(image_path).to(DEVICE)  # (1, 3, H, W)

    # ── 2. Build background (Gaussian noise around a black baseline) ──
    background = torch.zeros(n_background, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    background += torch.randn_like(background) * 0.1   # small noise

    # ── 3. Run SHAP GradientExplainer ─────────────────────────────────
    explainer   = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(input_tensor, nsamples=n_steps)
    # shap_values: list of arrays [class_0, class_1], each (1, 3, H, W)

    # ── 4. Predicted class ────────────────────────────────────────────
    with torch.no_grad():
        probs     = wrapped(input_tensor)
        pred_idx  = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()
    label_name = "Authentic" if pred_idx == 0 else "Counterfeit"

    # ── 5. Build per-pixel importance map ─────────────────────────────
    shap_arr = shap_values[0][0]         # (3, H, W) for predicted class
    # Aggregate across colour channels → (H, W)
    shap_2d  = shap_arr.mean(axis=0)

    # Per-channel absolute maps for separate visualisation
    shap_rgb = np.abs(shap_arr).transpose(1, 2, 0)   # (H, W, 3)
    shap_rgb = shap_rgb / (shap_rgb.max() + 1e-8)     # normalise to [0,1]

    # Original image (denormalised)
    orig_np = denormalise_tensor(input_tensor.cpu().squeeze(0))

    # ── 6. Plot ───────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0d1117")

    def _set_ax(ax, title):
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")

    # Row 0 ── overview
    axes[0, 0].imshow(orig_np)
    _set_ax(axes[0, 0], "Original Image")

    abs_max = max(abs(shap_2d.min()), abs(shap_2d.max())) + 1e-8
    im1 = axes[0, 1].imshow(shap_2d, cmap="coolwarm", vmin=-abs_max, vmax=abs_max)
    _set_ax(axes[0, 1], f"SHAP Values — '{label_name}'\n(warm=supports, cool=opposes)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    # Overlay: SHAP heatmap blended with original
    heatmap_norm = (shap_2d - shap_2d.min()) / (shap_2d.max() - shap_2d.min() + 1e-8)
    heatmap_rgb  = cm.coolwarm(heatmap_norm)[..., :3]         # (H, W, 3)
    heatmap_rgb = cv2.resize(heatmap_rgb, (224, 224))
    blended = (orig_np / 255.0 * 0.55 + heatmap_rgb * 0.45)
    blended      = np.clip(blended, 0, 1)
    axes[0, 2].imshow(blended)
    _set_ax(axes[0, 2], "Overlay — Original + SHAP Heatmap")

    # Row 1 ── per-channel importance
    channels = ["Red Channel", "Green Channel", "Blue Channel"]
    cmaps    = ["Reds", "Greens", "Blues"]
    for c_idx in range(3):
        ch_shap = np.abs(shap_arr[c_idx])
        im = axes[1, c_idx].imshow(ch_shap, cmap=cmaps[c_idx])
        _set_ax(axes[1, c_idx], f"SHAP |values| — {channels[c_idx]}")
        plt.colorbar(im, ax=axes[1, c_idx], fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    fig.suptitle(
        f"SHAP Explanation  |  Predicted: {label_name}  ({pred_prob*100:.1f}% confidence)",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"SHAP explanation saved → {output_path}")
    return output_path