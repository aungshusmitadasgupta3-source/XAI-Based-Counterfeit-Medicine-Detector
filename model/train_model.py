import argparse
import os
import sys
import copy
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from torchvision.models import EfficientNet_B0_Weights
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocessing import train_transform, inference_transform
CLASS_NAMES = ["authentic", "counterfeit"]
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model
def get_dataloaders(data_dir: str, batch_size: int = 32):
    """Build train/val ImageFolder DataLoaders."""
    data_dir = Path(data_dir)
    image_datasets = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val":   datasets.ImageFolder(data_dir / "val",   transform=inference_transform),
    }
    dataloaders = {
        phase: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
        )
        for phase, ds in image_datasets.items()
    }
    print(f"  Train samples : {len(image_datasets['train'])}")
    print(f"  Val   samples : {len(image_datasets['val'])}")
    print(f"  Classes       : {image_datasets['train'].classes}")
    return dataloaders, image_datasets["train"].classes
def train_model(
    model: nn.Module,
    dataloaders: dict,
    num_epochs: int = 20,
    lr: float = 1e-3,
    save_path: str = "model/model_weights.pth",
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_weights = copy.deepcopy(model.state_dict())
    best_acc     = 0.0
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  Training on : {DEVICE}")
    print(f"  Epochs      : {num_epochs}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        t0 = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc  = running_correct / len(dataloaders[phase].dataset)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"  {phase.capitalize():5s} | Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            if phase == "val" and epoch_acc > best_acc:
                best_acc     = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
        scheduler.step()
        print(f"  Elapsed: {time.time()-t0:.1f}s | Best val acc: {best_acc:.4f}\n")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, save_path)
    print(f"\n Model saved → {save_path}  (best val acc: {best_acc:.4f})")
    hist_path = Path(save_path).parent / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved → {hist_path}")
    model.load_state_dict(best_weights)
    return model, history
def main():
    parser = argparse.ArgumentParser(description="Train Counterfeit Medicine Detector")
    parser.add_argument("--data_dir",   default="data",                    help="Root dataset directory")
    parser.add_argument("--save_path",  default="model/model_weights.pth", help="Where to save weights")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--freeze",     action="store_true", help="Freeze backbone initially")
    args = parser.parse_args()
    print("\nCounterfeit Medicine Detection — Training Pipeline")
    dataloaders, classes = get_dataloaders(args.data_dir, args.batch_size)
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=args.freeze)
    print(f"  Backbone: EfficientNet-B0 | Classes: {classes}")
    train_model(model, dataloaders, args.epochs, args.lr, args.save_path)
if __name__ == "__main__":
    main()