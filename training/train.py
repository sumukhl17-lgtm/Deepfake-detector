from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.forensic import load_forensic_extractor
from models.fusion import FusionMLP
from training.dataset import CachedFeatureDataset


EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 5


def run_epoch(
    forensic: nn.Module,
    fusion: FusionMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    forensic.train(training)
    fusion.train(training)
    total_loss = 0.0
    correct = 0
    count = 0

    for spatial, frequency, raw_crop, labels in loader:
        spatial = spatial.to(device)
        frequency = frequency.to(device)
        raw_crop = raw_crop.to(device)
        labels = labels.to(device).unsqueeze(1)
        if training:
            optimizer.zero_grad()

        forensic_features = forensic(raw_crop)
        fused = torch.cat([spatial, frequency, forensic_features], dim=1)
        logits = fusion.logits(fused)
        loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        correct += ((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item()
        count += batch_size

    return total_loss / count, correct / count


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader = DataLoader(
        CachedFeatureDataset("train"), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        CachedFeatureDataset("val"), batch_size=BATCH_SIZE, shuffle=False
    )
    forensic = load_forensic_extractor().to(device)
    fusion = FusionMLP().to(device)
    trainable_parameters = [
        parameter
        for module in (forensic, fusion)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=1e-4
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_dir = PROJECT_ROOT / "models" / "weights"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            forensic, fusion, train_loader, loss_fn, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                forensic, fusion, val_loader, loss_fn, device, None
            )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch:02d}: train loss={train_loss:.4f}, "
            f"train accuracy={train_accuracy:.4f}, val loss={val_loss:.4f}, "
            f"val accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(forensic.state_dict(), checkpoint_dir / "forensic_cnn.pth")
            torch.save(fusion.state_dict(), checkpoint_dir / "fusion_mlp.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    epochs_run = range(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    loss_axis.plot(epochs_run, history["train_loss"], label="train")
    loss_axis.plot(epochs_run, history["val_loss"], label="val")
    loss_axis.set_title("Loss")
    loss_axis.legend()
    accuracy_axis.plot(epochs_run, history["train_accuracy"], label="train")
    accuracy_axis.plot(epochs_run, history["val_accuracy"], label="val")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.legend()
    figure.tight_layout()
    curve_path = results_dir / "training_curves.png"
    figure.savefig(curve_path)
    plt.close(figure)
    print(f"Saved {checkpoint_dir / 'forensic_cnn.pth'}")
    print(f"Saved {checkpoint_dir / 'fusion_mlp.pth'}")
    print(f"Saved {curve_path}")


if __name__ == "__main__":
    main()
