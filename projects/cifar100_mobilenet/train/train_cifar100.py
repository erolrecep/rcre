#!/usr/bin/env python3
"""
train_cifar100.py: Training and Export Pipeline for MobileNet on CIFAR-100
"""

import os
import argparse
from pathlib import Path
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model import MobileNetCIFAR100, export_mobilenet_to_rcre_binary

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_transform, test_transform

def export_test_samples(test_set, data_dir: Path, num_samples: int = 10, model: nn.Module = None, device: str = "cpu"):
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "samples.txt"
    manifest_lines = []

    print(f"📦 Extracting {num_samples} test samples to {data_dir}...")

    # Load class labels
    classes = test_set.classes if hasattr(test_set, "classes") else [f"class_{i}" for i in range(100)]

    for idx in range(num_samples):
        img_tensor, label = test_set[idx] # [3, 32, 32]
        sample_bin = data_dir / f"sample_{idx}.bin"

        # Save as contiguous float32 [3, 32, 32]
        arr = img_tensor.numpy().astype(np.float32)
        with open(sample_bin, "wb") as f:
            f.write(arr.tobytes())

        py_pred_idx = -1
        py_prob = 0.0
        if model is not None:
            model.eval()
            with torch.no_grad():
                inp = img_tensor.unsqueeze(0).to(device)
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)
                py_prob, py_pred = torch.topk(probs, 1)
                py_pred_idx = py_pred.item()
                py_prob = py_prob.item()

        class_name = classes[label]
        manifest_lines.append(f"{idx} {label} {class_name} {py_pred_idx} {py_prob:.4f}")
        print(f"  Sample {idx}: GT={label} ({class_name}) | PyTorch Top-1={py_pred_idx} ({classes[py_pred_idx]}) prob={py_prob*100:.1f}%")

    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print(f"✅ Samples and manifest written to {manifest_path}")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            _, pred5 = outputs.topk(5, 1, True, True)
            targets_view = targets.view(-1, 1).expand_as(pred5)
            correct_matrix = pred5.eq(targets_view)

            top1_correct += correct_matrix[:, 0].sum().item()
            top5_correct += correct_matrix.sum().item()
            total += targets.size(0)

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    return running_loss / total, top1_acc, top5_acc

CANONICAL_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm"
]

class SyntheticCIFAR(torch.utils.data.Dataset):
    def __init__(self, size=500):
        self.size = size
        self.classes = CANONICAL_CLASSES
        torch.manual_seed(42)
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = [i % 100 for i in range(size)]
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def main():
    parser = argparse.ArgumentParser(description="Train MobileNet on CIFAR-100 and Export to RCRE")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate")
    parser.add_argument("--export-only", action="store_true", help="Instantiate, export weights and samples without training")
    parser.add_argument("--download", action="store_true", help="Attempt to download CIFAR-100 dataset over the network")
    parser.add_argument("--synthetic", action="store_true", help="Use local synthetic CIFAR-100 dataset for fast training/testing")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-existing .pt checkpoint")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent.parent), help="Output directory for binary model & data")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using compute device: {device}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_bin_path = out_dir / "mobilenet_cifar100.bin"
    data_dir = out_dir / "data"

    print("🧠 Initializing MobileNet adapted for CIFAR-100...")
    model = MobileNetCIFAR100(num_classes=100).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture parameters: {num_params:,} ({num_params * 4 / (1024*1024):.2f} MB float32)")

    train_trans, test_trans = get_transforms()
    raw_data_dir = Path.home() / ".cache" / "torch_cifar100"

    cifar_exists = (raw_data_dir / "cifar-100-python.tar.gz").exists()
    if (args.download or cifar_exists) and not args.synthetic:
        try:
            print(f"📥 Loading CIFAR-100 dataset from {raw_data_dir}...")
            test_set = torchvision.datasets.CIFAR100(root=str(raw_data_dir), train=False, download=True, transform=test_trans)
            train_set = torchvision.datasets.CIFAR100(root=str(raw_data_dir), train=True, download=True, transform=train_trans)
        except Exception as e:
            print(f"⚠️ Network error downloading CIFAR-100: {e}. Switching to synthetic dataset...")
            train_set = SyntheticCIFAR(1000)
            test_set = SyntheticCIFAR(100)
    else:
        print("💡 Using synthetic CIFAR-100 dataset (instant local generation)...")
        train_set = SyntheticCIFAR(1000)
        test_set = SyntheticCIFAR(100)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📂 Loading weights from checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    elif not args.export_only and args.epochs > 0:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        print(f"🔥 Commencing training for {args.epochs} epoch(s)...")
        for ep in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, top1, top5 = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            print(f"   [Epoch {ep}/{args.epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Top-1: {top1:.2f}% Top-5: {top5:.2f}%")

    # Export Model to RCRE binary
    print(f"💾 Fusing BatchNorm layers and exporting model to {model_bin_path}...")
    export_mobilenet_to_rcre_binary(model, str(model_bin_path))

    # Export test sample binaries
    export_test_samples(test_set, data_dir, num_samples=10, model=model, device=device)

    print("🎉 MobileNet CIFAR-100 Training & Export Completed Successfully!")

if __name__ == "__main__":
    main()
