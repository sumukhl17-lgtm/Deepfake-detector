from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


KERNEL_PATH = Path(__file__).resolve().parents[1] / "models" / "srm_kernels.npy"


class ForensicExtractor(nn.Module):
    def __init__(self, kernels: np.ndarray) -> None:
        super().__init__()
        kernel_tensor = torch.from_numpy(kernels).float()
        self.srm = nn.Conv2d(
            in_channels=kernel_tensor.shape[1],
            out_channels=kernel_tensor.shape[0],
            kernel_size=kernel_tensor.shape[-2:],
            padding="same",
            bias=False,
        )
        with torch.no_grad():
            self.srm.weight.copy_(kernel_tensor)
        for parameter in self.srm.parameters():
            parameter.requires_grad = False
        self.cnn = nn.Sequential(
            nn.Conv2d(kernel_tensor.shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, face_crop: Tensor) -> Tensor:
        if face_crop.shape[1] == 3:
            face_crop = (
                face_crop[:, 0:1] * 0.299
                + face_crop[:, 1:2] * 0.587
                + face_crop[:, 2:3] * 0.114
            )
        return self.cnn(self.srm(face_crop)).flatten(1)


def load_forensic_extractor() -> ForensicExtractor:
    kernels = np.load(KERNEL_PATH)
    extractor = ForensicExtractor(kernels)
    return extractor


def extract_forensic_features(model: ForensicExtractor, face_crop: Tensor) -> Tensor:
    if face_crop.ndim == 3:
        face_crop = face_crop.unsqueeze(0)
    if face_crop.ndim != 4 or tuple(face_crop.shape[-2:]) != (224, 224):
        raise ValueError("face_crop must have shape (3, 224, 224) or (N, 3, 224, 224)")
    face_crop = face_crop.float()
    if face_crop.max() > 1:
        face_crop = face_crop / 255.0
    with torch.inference_mode():
        features = model(face_crop)
    return features.squeeze(0) if features.shape[0] == 1 else features
