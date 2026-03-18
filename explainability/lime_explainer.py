"""
explainability/lime_explainer.py
---------------------------------
LIME (Local Interpretable Model-Agnostic Explanations) for the
Counterfeit Medicine Detection System.

How LIME works
--------------
LIME explains a single prediction by:
1. Perturbing the input image into ~1000 variants by randomly masking
   superpixel segments (patches of similar pixels).
2. Running every variant through the black-box model to collect predictions.
3. Fitting a lightweight interpretable model (weighted ridge regression)
   on those (perturbation, prediction) pairs — weighted by proximity to
   the original image.
4. The coefficients of that linear surrogate reveal which superpixels
   contributed MOST to the prediction.

In medicine-packaging terms this surfaces regions such as logo blocks,
barcode strips, or text alignment zones that pushed the model toward
"Authentic" or "Counterfeit."
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocessing import (
    preprocess_numpy_for_lime,
    batch_numpy_to_tensor,
    IMG_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def generate_lime_explanation(
    model: torch.nn.Module,
    image_path: str,
    output_path: str = "outputs/lime_explanation.png",
    num_samples: int = 1000,
    num_features: int = 10,
    positive_only: bool = True,
    hide_rest: bool = False,
) -> str:
    """
    Generate a LIME explanation image for a medicine package photo.

    Parameters
    ----------
    model        : trained PyTorch model (eval mode)
    image_path   : path to the input image
    output_path  : where to save the PNG
    num_samples  : LIME perturbation count (higher → more accurate, slower)
    num_features : number of superpixel regions to highlight
    positive_only: show only regions that support the predicted class
    hide_rest    : grey out non-highlighted regions

    Returns
    -------
    output_path  : path of the saved PNG
    """
    model.eval()
    model.to(DEVICE)

    # ── 1. Load image as uint8 numpy (H, W, 3) ───────────────────────
    img_np = preprocess_numpy_for_lime(image_path)

    # ── 2. Build LIME batch-predict callback ──────────────────────────
    def batch_predict(images: np.ndarray) -> np.ndarray:
        """
        LIME calls this with shape (N, H, W, 3) float64 in [0,1].
        We must return (N, num_classes) probability array.
        """
        images_uint8 = (images * 255).astype(np.uint8)
        tensors = batch_numpy_to_tensor(images_uint8).to(DEVICE)
        with torch.no_grad():
            logits = model(tensors)
            probs  = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # ── 3. Run LIME ───────────────────────────────────────────────────
    explainer = lime_image.LimeImageExplainer(random_state=42)

    explanation = explainer.explain_instance(
        image          = img_np.astype(np.float64) / 255.0,
        classifier_fn  = batch_predict,
        top_labels     = 2,
        hide_color     = 0,
        num_samples    = num_samples,
        segmentation_fn= None,   # default: Quickshift
    )

    # ── 4. Get predicted label ────────────────────────────────────────
    top_label = explanation.top_labels[0]
    label_name = "Authentic" if top_label == 0 else "Counterfeit"

    # ── 5. Build explanation overlay ─────────────────────────────────
    temp_img, mask = explanation.get_image_and_mask(
        label        = top_label,
        positive_only= positive_only,
        num_features = num_features,
        hide_rest    = hide_rest,
    )

    # ── 6. Plot & save ────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")

    # --- Original ---
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    # --- LIME overlay ---
    overlay = mark_boundaries(
        temp_img / 255.0 if temp_img.max() > 1 else temp_img,
        mask, color=(1, 0.6, 0), mode="thick"
    )
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"LIME — Regions supporting\n'{label_name}'",
        color="#f0a500", fontsize=13, pad=10,
    )
    axes[1].axis("off")

    # --- Heatmap from superpixel weights ---
    ind_map = explanation.local_exp[top_label]
    seg     = explanation.segments
    heat    = np.zeros(seg.shape, dtype=np.float32)
    for seg_id, weight in ind_map:
        heat[seg == seg_id] = weight
    # normalise to [-1, 1]
    abs_max = max(abs(heat.min()), abs(heat.max())) + 1e-8
    heat    = heat / abs_max

    im = axes[2].imshow(heat, cmap="RdYlGn", vmin=-1, vmax=1, interpolation="bilinear")
    axes[2].set_title("Superpixel Weight Map\n(green=supports, red=opposes)", color="white", fontsize=13, pad=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(plt.colorbar(im, ax=axes[2], fraction=0, pad=0).ax, "yticklabels"), color="white")

    fig.suptitle(
        f"LIME Explanation  |  Predicted: {label_name}",
        color="white", fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"LIME explanation saved → {output_path}")
    return output_path