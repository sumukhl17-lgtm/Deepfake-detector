from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.forensic import load_forensic_extractor
from models.fusion import FusionMLP
from training.dataset import CachedFeatureDataset


BATCH_SIZE = 32


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir = PROJECT_ROOT / "models" / "weights"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    forensic = load_forensic_extractor().to(device)
    forensic.load_state_dict(
        torch.load(
            checkpoint_dir / "forensic_cnn.pth",
            map_location=device,
            weights_only=True,
        )
    )
    forensic.eval()

    fusion = FusionMLP().to(device)
    fusion.load_state_dict(
        torch.load(
            checkpoint_dir / "fusion_mlp.pth",
            map_location=device,
            weights_only=True,
        )
    )
    fusion.eval()

    test_loader = DataLoader(
        CachedFeatureDataset("test"),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    actual: list[int] = []
    predicted: list[int] = []

    with torch.inference_mode():
        for spatial, frequency, raw_crop, labels in test_loader:
            fused = torch.cat(
                [
                    spatial.to(device),
                    frequency.to(device),
                    forensic(raw_crop.to(device)),
                ],
                dim=1,
            )
            probabilities = fusion(fused).squeeze(1)
            predicted.extend((probabilities >= 0.5).long().cpu().tolist())
            actual.extend(labels.long().tolist())

    accuracy = accuracy_score(actual, predicted)
    fake_precision = precision_score(
        actual, predicted, pos_label=1, zero_division=0
    )
    fake_recall = recall_score(actual, predicted, pos_label=1, zero_division=0)
    fake_f1 = f1_score(actual, predicted, pos_label=1, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["real", "fake"],
        zero_division=0,
    )

    print(f"Test examples: {len(actual)}")
    print(f"Device: {device}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Fake precision: {fake_precision:.4f}")
    print(f"Fake recall: {fake_recall:.4f}")
    print(f"Fake F1-score: {fake_f1:.4f}")
    print("Confusion matrix (rows = actual, columns = predicted; [real, fake]):")
    print(matrix)
    print("Classification report:")
    print(report)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Test Confusion Matrix",
    )
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    confusion_matrix_path = results_dir / "confusion_matrix.png"
    figure.savefig(confusion_matrix_path)
    plt.close(figure)

    metrics = {
        "test_examples": len(actual),
        "accuracy": accuracy,
        "fake": {
            "precision": fake_precision,
            "recall": fake_recall,
            "f1_score": fake_f1,
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        ),
    }
    metrics_path = results_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved confusion matrix: {confusion_matrix_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
