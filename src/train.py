import os
import sys
import json
import time
import copy
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Crop Dataset"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "cropguard_best.pth"
METADATA_PATH = MODEL_DIR / "metadata.json"

# ── Hyper-parameters ───────────────────────────────────────────────────────────
IMG_SIZE = 160          # 224→160 cuts ~2x compute per batch
BATCH_SIZE = 64         # larger batch = fewer gradient steps per epoch
FREEZE_EPOCHS = 5       # phase 1: train classifier head only (very fast)
FINETUNE_EPOCHS = 10    # phase 2: unfreeze whole network, low LR
NUM_EPOCHS = FREEZE_EPOCHS + FINETUNE_EPOCHS
LR_HEAD = 1e-3          # higher LR when backbone is frozen
LR_FINETUNE = 5e-5      # low LR when fine-tuning full network
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
SEED = 42
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_datasets():
    train_tf, val_tf = get_transforms()
    full = datasets.ImageFolder(str(DATASET_DIR), transform=train_tf)
    classes = full.classes
    n = len(full)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val], generator=generator)

    # Apply val transform to the val split via a wrapper
    val_ds.dataset = copy.deepcopy(full)
    val_ds.dataset.transform = val_tf

    # Weighted sampler to handle class imbalance in training
    targets = [full.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets, minlength=len(classes))
    weights = 1.0 / class_counts
    sample_weights = [weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0)
    return train_loader, val_loader, classes


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def freeze_backbone(model: nn.Module):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


def plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def main():
    seed_everything(SEED)
    print("Loading dataset …")
    train_loader, val_loader, classes = build_datasets()
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = build_model(len(classes))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Phase 1: frozen backbone, train head only
    print(f"\n-- Phase 1: frozen backbone ({FREEZE_EPOCHS} epochs) --")
    freeze_backbone(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS, eta_min=1e-5)

    for epoch in range(1, FREEZE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        elapsed = time.time() - t0
        print(f"[P1] Epoch {epoch:02d}/{FREEZE_EPOCHS}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}  {elapsed:.1f}s")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [saved] best model (val_acc={vl_acc:.4f})")

    # Phase 2: unfreeze all, fine-tune with small LR
    print(f"\n-- Phase 2: full fine-tuning ({FINETUNE_EPOCHS} epochs) --")
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        elapsed = time.time() - t0
        print(f"[P2] Epoch {epoch:02d}/{FINETUNE_EPOCHS}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}  {elapsed:.1f}s")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [saved] best model (val_acc={vl_acc:.4f})")

    # Save metadata
    metadata = {
        "classes": classes,
        "num_classes": len(classes),
        "img_size": IMG_SIZE,
        "best_val_acc": best_val_acc,
        "architecture": "efficientnet_b0",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved → {METADATA_PATH}")

    # Final evaluation
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    _, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion)
    print(f"\nFinal val accuracy (best model): {final_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=classes))

    cm = confusion_matrix(final_labels, final_preds)
    plot_confusion_matrix(cm, classes, MODEL_DIR / "confusion_matrix.png")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, NUM_EPOCHS + 1)
    ax1.plot(epochs_range, history["train_loss"], label="Train")
    ax1.plot(epochs_range, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epochs_range, history["train_acc"], label="Train")
    ax2.plot(epochs_range, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_curves.png", dpi=100)
    plt.close()
    print(f"Training curves saved → {MODEL_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
