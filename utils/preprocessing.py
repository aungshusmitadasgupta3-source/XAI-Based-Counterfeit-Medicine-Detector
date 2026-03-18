"""
utils/preprocessing.py
-----------------------
Image preprocessing utilities for the Counterfeit Medicine Detection System.
Handles loading, resizing, normalizing, and converting images for the model.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = 224          # EfficientNet / ResNet input size
MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
STD  = [0.229, 0.224, 0.225]   # ImageNet std

# ─────────────────────────────────────────────
# Core transforms
# ─────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_image_pil(image_path: str) -> Image.Image:
    """Load an image from disk and convert to RGB PIL Image."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(path).convert("RGB")
    return img


def load_image_cv2(image_path: str) -> np.ndarray:
    """Load an image with OpenCV in RGB format."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preprocess_for_inference(image_path: str) -> torch.Tensor:
    """
    Load an image and apply inference transforms.
    Returns a (1, 3, H, W) tensor.
    """
    img = load_image_pil(image_path)
    tensor = inference_transform(img)
    return tensor.unsqueeze(0)   # add batch dimension


def preprocess_numpy_for_lime(image_path: str) -> np.ndarray:
    """
    Load and resize an image for LIME.
    LIME expects a uint8 RGB numpy array of shape (H, W, 3).
    """
    img = load_image_cv2(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img_resized   # uint8, RGB, 0-255


def numpy_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """
    Convert a float32 or uint8 numpy image (H, W, 3) -> normalised (1, 3, H, W) tensor.
    Used inside LIME's batch_predict callback.
    """
    # Ensure float in [0, 1]
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0

    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    tensor = inference_transform(img_pil)
    return tensor


def batch_numpy_to_tensor(images: np.ndarray) -> torch.Tensor:
    """
    Convert a batch of numpy images (N, H, W, 3) -> normalised (N, 3, H, W) tensor.
    Used by LIME's batch_predict function.
    """
    tensors = [numpy_to_tensor(img) for img in images]
    return torch.stack(tensors)


def denormalise_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation for visualisation.
    Input : (C, H, W) or (1, C, H, W) tensor
    Output: uint8 numpy array (H, W, 3)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)