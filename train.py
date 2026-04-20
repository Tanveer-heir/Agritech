"""
train.py — EfficientNetV2-S Fine-Tuning on PlantVillage
========================================================
Crop Doctor Project | Run locally on RTX A4000

Usage:
    python train.py                          # uses defaults
    python train.py --data ./data/plantvillage --epochs 30
    python train.py --phase1-only            # just train the head, skip fine-tune
    python train.py --resume ./models/phase1_best.pth

Requirements (no admin needed — pure pip):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm scikit-learn matplotlib seaborn tqdm pillow

Output files (saved to ./models/):
    efficientnetv2_crop_doctor.pth   — full model weights
    class_mapping.json               — class index → name + metadata
    training_curves.png              — accuracy / loss plots
    confusion_matrix.png             — test set confusion matrix

Architecture:
    Photo → EfficientNetV2-S (timm) → GlobalAvgPool → BN → Dropout → Dense(512) → Dense(N)
    If confidence < CONFIDENCE_THRESHOLD → escalate to Gemini Vision
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ── timm: pip install timm (no admin needed) ──────────────────────────────────
try:
    import timm
except ImportError:
    raise SystemExit(
        "timm not found. Run:\n"
        "  pip install timm\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Paths
    "DATA_DIR":    "./data/plantvillage",
    "OUTPUT_DIR":  "./models",

    # Image
    "IMG_SIZE":    224,       # EfficientNetV2-S native size
    "BATCH_SIZE":  64,        # A4000 has 16 GB VRAM — 64 fits easily; bump to 128 if you want

    # Splits
    "VAL_SPLIT":   0.15,
    "TEST_SPLIT":  0.10,

    # Phase 1 — head only, base frozen
    "PHASE1_EPOCHS": 10,
    "PHASE1_LR":     1e-3,

    # Phase 2 — fine-tune top layers
    "PHASE2_EPOCHS":   25,
    "PHASE2_LR":       1e-4,
    "UNFREEZE_BLOCKS": 3,     # unfreeze last N blocks of EfficientNetV2-S

    # Regularisation
    "DROPOUT":          0.3,
    "LABEL_SMOOTHING":  0.1,
    "WEIGHT_DECAY":     1e-4,

    # Production threshold
    "CONFIDENCE_THRESHOLD": 0.85,

    # Mixed precision (RTX A4000 fully supports bfloat16 / float16)
    "AMP": True,

    # Reproducibility
    "SEED": 42,

    # Workers — A4000 workstation typically has many cores
    "NUM_WORKERS": 8,
}


# ══════════════════════════════════════════════════════════════════════════════
# SEEDS & DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # keep fast on A4000
    torch.backends.cudnn.benchmark = True        # auto-tune convs for your GPU


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU  : {name}  ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("WARNING: No GPU found — training will be very slow on CPU.")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PlantVillageDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_dataset(data_dir: str) -> tuple[list, list, list, int]:
    """
    Walk data_dir/<ClassName>/*.jpg structure.
    Returns (all_paths, all_labels, class_names, num_classes).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'.\n"
            "Download with:\n"
            "  pip install kaggle\n"
            "  kaggle datasets download -d emmarex/plantdisease -p ./data --unzip\n"
        )

    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"No subdirectories found in '{data_dir}'. "
                         "Expected one folder per class.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    all_paths, all_labels = [], []

    for cls in class_names:
        cls_dir = data_dir / cls
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
            for p in cls_dir.glob(ext):
                all_paths.append(str(p))
                all_labels.append(class_to_idx[cls])

    print(f"\nDataset: {data_dir}")
    print(f"  Classes : {len(class_names)}")
    print(f"  Images  : {len(all_paths):,}")
    return all_paths, all_labels, class_names, len(class_names)


def make_splits(paths, labels, val_split=0.15, test_split=0.10, seed=42):
    """Stratified train / val / test split."""
    train_val_p, test_p, train_val_l, test_l = train_test_split(
        paths, labels,
        test_size=test_split,
        stratify=labels,
        random_state=seed,
    )
    relative_val = val_split / (1.0 - test_split)
    train_p, val_p, train_l, val_l = train_test_split(
        train_val_p, train_val_l,
        test_size=relative_val,
        stratify=train_val_l,
        random_state=seed,
    )
    print(f"\nSplit  train={len(train_p):,}  val={len(val_p):,}  test={len(test_p):,}")
    return train_p, val_p, test_p, train_l, val_l, test_l


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CropDoctorNet(nn.Module):
    """
    EfficientNetV2-S backbone (timm) with custom classification head.

    Head: GlobalAvgPool → BN → Dropout(0.3) → Dense(512, ReLU) → Dropout(0.15) → Dense(N)
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # timm model — downloads ImageNet weights automatically on first run
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            num_classes=0,           # remove timm's default head
            global_pool="avg",       # built-in global average pooling
        )
        feature_dim = self.backbone.num_features  # 1280 for V2-S

        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # shape: (B, 1280)
        return self.head(features)    # shape: (B, num_classes)

    def freeze_backbone(self) -> None:
        """Phase 1: only train the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — training head only.")

    def unfreeze_top_blocks(self, n_blocks: int = 3) -> None:
        """
        Phase 2: unfreeze the last n_blocks of the EfficientNetV2 backbone.
        EfficientNetV2-S has blocks named blocks.0 … blocks.5 + conv_head.
        We unfreeze from the end.
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then selectively unfreeze
        # timm EfficientNetV2 structure: backbone.blocks[i] for i in 0..5, + conv_head, bn2
        blocks = list(self.backbone.blocks.children())   # list of block groups
        unfreeze_from = len(blocks) - n_blocks

        for i, block in enumerate(blocks):
            if i >= unfreeze_from:
                for param in block.parameters():
                    param.requires_grad = True

        # Always unfreeze conv_head and bn2 (final feature layers)
        for name in ("conv_head", "bn2"):
            layer = getattr(self.backbone, name, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze last {n_blocks} blocks — trainable params: {n_trainable:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    """One epoch of train or validation. Returns (avg_loss, accuracy)."""
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for imgs, labels in tqdm(loader, leave=False, desc="train" if training else "val "):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=CFG["AMP"]):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train_phase(
    model, train_loader, val_loader, optimizer, scheduler,
    criterion, scaler, device, epochs, phase_name, output_dir,
):
    best_val_acc = 0.0
    best_path    = output_dir / f"{phase_name}_best.pth"
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, training=True)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, training=False)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        marker  = " ★" if val_acc > best_val_acc else ""
        print(
            f"[{phase_name}] Epoch {epoch:>3}/{epochs}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"{elapsed:.0f}s{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"\n✅ {phase_name} complete — best val_accuracy: {best_val_acc:.4f}")
    print(f"   Weights saved: {best_path}")

    # Restore best weights for next phase / evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    return history


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_set(model, test_loader, device, class_names, output_dir):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in tqdm(test_loader, desc="test eval"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=CFG["AMP"]):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_labels.extend(labels.numpy().tolist())

    all_probs  = np.concatenate(all_probs, axis=0)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc    = (all_preds == all_labels).mean()
    top3   = np.sum([l in np.argsort(p)[-3:] for l, p in zip(all_labels, all_probs)]) / len(all_labels)
    max_p  = all_probs.max(axis=1)

    # Escalation stats at threshold
    thresh    = CFG["CONFIDENCE_THRESHOLD"]
    low_conf  = max_p < thresh
    esc_pct   = low_conf.mean() * 100
    eff_acc   = (all_preds[~low_conf] == all_labels[~low_conf]).mean() * 100

    print(f"\n{'='*55}")
    print(f"TEST RESULTS")
    print(f"{'='*55}")
    print(f"  Top-1 accuracy    : {acc*100:.2f}%")
    print(f"  Top-3 accuracy    : {top3*100:.2f}%")
    print(f"  At threshold={thresh}:")
    print(f"    EfficientNet handles : {(~low_conf).sum():,} ({100-esc_pct:.1f}%)")
    print(f"    Escalated to Gemini  : {low_conf.sum():,} ({esc_pct:.1f}%)")
    print(f"    EfficientNet acc (high-conf): {eff_acc:.2f}%")
    print(f"{'='*55}")

    # Confusion matrix (use short names)
    short_names = []
    for n in class_names:
        parts   = n.split("___")
        crop    = parts[0].replace("_", " ")
        disease = (parts[1].replace("_", " ") if len(parts) > 1 else "Healthy")[:18]
        short_names.append(f"{crop}\n{disease}")

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm_norm, annot=False, cmap="YlOrRd",
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.2, ax=ax, vmin=0, vmax=1)
    ax.set_title("Normalised Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {output_dir}/confusion_matrix.png")

    # Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    correct = all_preds == all_labels
    ax.hist(max_p[correct],  bins=60, alpha=0.7, color="#1A6B3C",
            label=f"Correct ({correct.sum():,})")
    ax.hist(max_p[~correct], bins=60, alpha=0.7, color="#8B1A1A",
            label=f"Incorrect ({(~correct).sum():,})")
    ax.axvline(thresh, color="#7A4F00", lw=2, linestyle="--",
               label=f"Threshold = {thresh}")
    ax.set_xlabel("Max softmax confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution — Correct vs Incorrect", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "confidence_distribution.png", dpi=150)
    plt.close()

    # Per-class report (last 20 classes to keep terminal clean)
    print("\nPer-class report (sample):")
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names, digits=3,
                                 output_dict=False)[:3000])

    return float(acc), float(top3), float(esc_pct), float(eff_acc)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(h1, h2, output_dir):
    acc   = h1["train_acc"]  + h2["train_acc"]
    vacc  = h1["val_acc"]    + h2["val_acc"]
    loss  = h1["train_loss"] + h2["train_loss"]
    vloss = h1["val_loss"]   + h2["val_loss"]
    p1_end = len(h1["train_acc"])
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tv, vv, title in [
        (ax1, acc, vacc, "Accuracy"),
        (ax2, loss, vloss, "Loss"),
    ]:
        ax.plot(epochs, tv, "#1A6B3C", lw=2, label="Train")
        ax.plot(epochs, vv, "#7A4F00", lw=2, linestyle="--", label="Validation")
        ax.axvline(p1_end, color="#9B9790", lw=1.5, linestyle=":", label="Phase 2 start")
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Training — Phase 1 (frozen head) + Phase 2 (fine-tune)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Training curves saved: {output_dir}/training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNetV2-S crop disease classifier")
    p.add_argument("--data",          default=CFG["DATA_DIR"],   help="Path to PlantVillage dataset root")
    p.add_argument("--output",        default=CFG["OUTPUT_DIR"], help="Directory to save models and plots")
    p.add_argument("--batch-size",    type=int, default=CFG["BATCH_SIZE"])
    p.add_argument("--phase1-epochs", type=int, default=CFG["PHASE1_EPOCHS"])
    p.add_argument("--phase2-epochs", type=int, default=CFG["PHASE2_EPOCHS"])
    p.add_argument("--unfreeze-blocks", type=int, default=CFG["UNFREEZE_BLOCKS"],
                   help="How many EfficientNet blocks to unfreeze in phase 2 (default 3)")
    p.add_argument("--threshold",     type=float, default=CFG["CONFIDENCE_THRESHOLD"],
                   help="Confidence below this → escalate to Gemini Vision")
    p.add_argument("--phase1-only",   action="store_true",
                   help="Stop after phase 1 (head only training)")
    p.add_argument("--resume",        default=None,
                   help="Path to .pth checkpoint to resume from (skips phase 1)")
    p.add_argument("--workers",       type=int, default=CFG["NUM_WORKERS"])
    p.add_argument("--seed",          type=int, default=CFG["SEED"])
    return p.parse_args()


def main():
    args = parse_args()

    # Update CFG from args
    CFG["DATA_DIR"]              = args.data
    CFG["OUTPUT_DIR"]            = args.output
    CFG["BATCH_SIZE"]            = args.batch_size
    CFG["PHASE1_EPOCHS"]         = args.phase1_epochs
    CFG["PHASE2_EPOCHS"]         = args.phase2_epochs
    CFG["UNFREEZE_BLOCKS"]       = args.unfreeze_blocks
    CFG["CONFIDENCE_THRESHOLD"]  = args.threshold
    CFG["NUM_WORKERS"]           = args.workers
    CFG["SEED"]                  = args.seed

    output_dir = Path(CFG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(CFG["SEED"])
    device = get_device()

    print("\n⚙️  Configuration:")
    for k, v in CFG.items():
        print(f"   {k:<26} = {v}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    all_paths, all_labels, class_names, num_classes = load_dataset(CFG["DATA_DIR"])
    train_p, val_p, test_p, train_l, val_l, test_l = make_splits(
        all_paths, all_labels,
        val_split=CFG["VAL_SPLIT"],
        test_split=CFG["TEST_SPLIT"],
        seed=CFG["SEED"],
    )

    train_ds = PlantVillageDataset(train_p, train_l, TRAIN_TRANSFORMS)
    val_ds   = PlantVillageDataset(val_p,   val_l,   VAL_TRANSFORMS)
    test_ds  = PlantVillageDataset(test_p,  test_l,  VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True,
        num_workers=CFG["NUM_WORKERS"], pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False,
        num_workers=CFG["NUM_WORKERS"], pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False,
        num_workers=CFG["NUM_WORKERS"], pin_memory=True, persistent_workers=True,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = CropDoctorNet(num_classes=num_classes, dropout=CFG["DROPOUT"]).to(device)
    print(f"\nModel: CropDoctorNet (EfficientNetV2-S backbone)")
    print(f"  Output classes : {num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params   : {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["LABEL_SMOOTHING"])
    scaler    = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    history1, history2 = {}, {}

    # ── Phase 1 — head only ───────────────────────────────────────────────────
    if args.resume:
        print(f"\n↩  Resuming from {args.resume} — skipping phase 1")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    else:
        print("\n" + "="*55)
        print("PHASE 1 — Training head only (backbone frozen)")
        print("="*55)
        model.freeze_backbone()

        optimizer1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CFG["PHASE1_LR"],
            weight_decay=CFG["WEIGHT_DECAY"],
        )
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, mode="min", factor=0.5, patience=2, verbose=True)

        history1 = train_phase(
            model, train_loader, val_loader,
            optimizer1, scheduler1, criterion, scaler, device,
            epochs=CFG["PHASE1_EPOCHS"],
            phase_name="phase1",
            output_dir=output_dir,
        )

    if args.phase1_only:
        print("\n--phase1-only flag set. Stopping after phase 1.")
        _save_outputs(model, class_names, num_classes, history1, {}, output_dir, 0.0, 0.0, 0.0, 0.0)
        return

    # ── Phase 2 — fine-tune top blocks ────────────────────────────────────────
    print("\n" + "="*55)
    print(f"PHASE 2 — Fine-tuning top {CFG['UNFREEZE_BLOCKS']} blocks")
    print("="*55)
    model.unfreeze_top_blocks(CFG["UNFREEZE_BLOCKS"])

    optimizer2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["PHASE2_LR"],
        weight_decay=CFG["WEIGHT_DECAY"],
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=CFG["PHASE2_EPOCHS"], eta_min=1e-7)

    history2 = train_phase(
        model, train_loader, val_loader,
        optimizer2, scheduler2, criterion, scaler, device,
        epochs=CFG["PHASE2_EPOCHS"],
        phase_name="phase2",
        output_dir=output_dir,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("EVALUATION — Held-out test set")
    print("="*55)
    test_acc, top3_acc, esc_pct, eff_acc = evaluate_test_set(
        model, test_loader, device, class_names, output_dir)

    # ── Save all outputs ──────────────────────────────────────────────────────
    _save_outputs(model, class_names, num_classes, history1, history2,
                  output_dir, test_acc, top3_acc, esc_pct, eff_acc)

    if history1 and history2:
        plot_training_curves(history1, history2, output_dir)

    print("\n🎉 Training complete!")
    print(f"\nFiles saved to {output_dir}/")
    print(f"  efficientnetv2_crop_doctor.pth  — model weights")
    print(f"  class_mapping.json              — class names + metadata")
    print(f"  training_curves.png             — accuracy / loss curves")
    print(f"  confusion_matrix.png            — test set confusion matrix")
    print(f"  confidence_distribution.png     — threshold analysis")
    print(f"\nNext: copy both .pth and class_mapping.json to your project root,")
    print(f"      then run vision_agent.py — it will load them automatically.")


def _save_outputs(model, class_names, num_classes, h1, h2,
                  output_dir, test_acc, top3_acc, esc_pct, eff_acc):
    # Save model weights
    model_path = output_dir / "efficientnetv2_crop_doctor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model weights saved: {model_path}")

    # Save class mapping — used by vision_agent.py
    class_mapping = {
        "class_names":           class_names,
        "class_to_idx":          {n: i for i, n in enumerate(class_names)},
        "num_classes":           num_classes,
        "img_size":              CFG["IMG_SIZE"],
        "confidence_threshold":  CFG["CONFIDENCE_THRESHOLD"],
        "model_file":            "efficientnetv2_crop_doctor.pth",
        "training_info": {
            "framework":           "PyTorch + timm",
            "backbone":            "tf_efficientnetv2_s",
            "phase1_epochs":       len(h1.get("train_acc", [])),
            "phase2_epochs":       len(h2.get("train_acc", [])),
            "best_val_accuracy":   max(h2.get("val_acc", [0])) if h2 else max(h1.get("val_acc", [0])),
            "test_accuracy":       test_acc,
            "top3_accuracy":       top3_acc,
            "escalation_rate_pct": esc_pct,
            "high_conf_accuracy":  eff_acc,
        },
    }

    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"✅ Class mapping saved : {mapping_path}")


if __name__ == "__main__":
    main()
