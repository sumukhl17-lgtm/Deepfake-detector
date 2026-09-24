from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import efficientnet_b0


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "weights"
    / "efficientnet_b0_ffpp"
    / "efficientnet_b0_ffpp_c23.pth"
)


class SpatialExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, face_tensor: Tensor) -> Tensor:
        normalized = (face_tensor - self.imagenet_mean) / self.imagenet_std
        features = self.features(normalized)
        return torch.flatten(self.avgpool(features), 1)


def load_spatial_extractor() -> SpatialExtractor:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state dict, got {type(checkpoint).__name__}")

    backbone = efficientnet_b0(weights=None)
    classifier_weight = checkpoint.get("classifier.1.weight")
    if not isinstance(classifier_weight, torch.Tensor) or classifier_weight.ndim != 2:
        raise RuntimeError("Checkpoint does not contain a valid classifier.1.weight")
    backbone.classifier[1] = nn.Linear(
        classifier_weight.shape[1], classifier_weight.shape[0]
    )
    incompatible = backbone.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    extractor = SpatialExtractor(backbone)
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    extractor.eval()
    return extractor


@torch.inference_mode()
def extract_spatial_features(model: SpatialExtractor, face_tensor: Tensor) -> Tensor:
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    if face_tensor.ndim != 4 or tuple(face_tensor.shape[-2:]) != (224, 224):
        raise ValueError("face_tensor must have shape (3, 224, 224) or (N, 3, 224, 224)")
    features = model(face_tensor)
    return features.squeeze(0) if features.shape[0] == 1 else features
