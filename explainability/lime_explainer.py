import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          
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
def generate_lime_explanation(
    model: torch.nn.Module,
    image_path: str,
    output_path: str = "outputs/lime_explanation.png",
    num_samples: int = 1000,
    num_features: int = 10,
    positive_only: bool = True,
    hide_rest: bool = False,
) -> str:
    model.eval()
    model.to(DEVICE)
    img_np = preprocess_numpy_for_lime(image_path)
    def batch_predict(images: np.ndarray) -> np.ndarray:
        images_uint8 = (images * 255).astype(np.uint8)
        tensors = batch_numpy_to_tensor(images_uint8).to(DEVICE)
        with torch.no_grad():
            logits = model(tensors)
            probs  = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image          = img_np.astype(np.float64) / 255.0,
        classifier_fn  = batch_predict,
        top_labels     = 2,
        hide_color     = 0,
        num_samples    = num_samples,
        segmentation_fn= None,   # default: Quickshift
    )
    top_label = explanation.top_labels[0]
    label_name = "Authentic" if top_label == 0 else "Counterfeit"
    temp_img, mask = explanation.get_image_and_mask(
        label        = top_label,
        positive_only= positive_only,
        num_features = num_features,
        hide_rest    = hide_rest,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")
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
    ind_map = explanation.local_exp[top_label]
    seg     = explanation.segments
    heat    = np.zeros(seg.shape, dtype=np.float32)
    for seg_id, weight in ind_map:
        heat[seg == seg_id] = weight
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
