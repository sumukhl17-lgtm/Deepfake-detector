from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(features))

    @torch.inference_mode()
    def fuse_and_classify(
        self,
        spatial_vec: Tensor,
        freq_vec: Tensor,
        forensic_vec: Tensor,
    ) -> tuple[str, float]:
        combined = torch.cat(
            [spatial_vec.flatten(), freq_vec.flatten(), forensic_vec.flatten()]
        ).float()
        if combined.shape != (1536,):
            raise ValueError(f"Expected 1536 fused features, got {tuple(combined.shape)}")
        confidence = float(self(combined.unsqueeze(0)).item())
        label = "FAKE" if confidence > 0.5 else "REAL"
        return label, confidence


def fuse_and_classify(
    spatial_vec: Tensor,
    freq_vec: Tensor,
    forensic_vec: Tensor,
    model: FusionMLP | None = None,
) -> tuple[str, float]:
    classifier = model if model is not None else FusionMLP()
    classifier.eval()
    return classifier.fuse_and_classify(spatial_vec, freq_vec, forensic_vec)
