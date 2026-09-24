from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import Tensor, nn

from models.forensic import extract_forensic_features, load_forensic_extractor
from models.frequency import extract_frequency_features
from models.fusion import FusionMLP
from models.spatial import SpatialExtractor, extract_spatial_features, load_spatial_extractor


class FusionGradCAMWrapper(nn.Module):
    def __init__(
        self,
        spatial_extractor: SpatialExtractor,
        forensic_extractor: nn.Module,
        fusion_model: FusionMLP,
    ) -> None:
        super().__init__()
        self.spatial_extractor = spatial_extractor
        self.forensic_extractor = forensic_extractor
        self.fusion_model = fusion_model

    def forward(self, face_crop: Tensor) -> Tensor:
        face_crop = face_crop.requires_grad_(True)
        normalized = (
            face_crop - self.spatial_extractor.imagenet_mean
        ) / self.spatial_extractor.imagenet_std
        spatial_map = self.spatial_extractor.features(normalized)
        spatial_vec = torch.flatten(
            self.spatial_extractor.avgpool(spatial_map), 1
        )
        with torch.no_grad():
            face_cpu = face_crop.detach().cpu()
            frequency_vec = torch.from_numpy(
                np.stack(
                    [extract_frequency_features(sample) for sample in face_cpu]
                )
            ).to(face_crop.device)
            forensic_vec = extract_forensic_features(
                self.forensic_extractor, face_cpu
            ).to(face_crop.device)
            if forensic_vec.ndim == 1:
                forensic_vec = forensic_vec.unsqueeze(0)
        fused = torch.cat([spatial_vec, frequency_vec, forensic_vec], dim=1)
        return self.fusion_model(fused)


class ForensicGradCAMWrapper(FusionGradCAMWrapper):
    def forward(self, face_crop: Tensor) -> Tensor:
        face_crop = face_crop.requires_grad_(True)
        with torch.no_grad():
            normalized = (
                face_crop - self.spatial_extractor.imagenet_mean
            ) / self.spatial_extractor.imagenet_std
            spatial_map = self.spatial_extractor.features(normalized)
            spatial_vec = torch.flatten(
                self.spatial_extractor.avgpool(spatial_map), 1
            )
            face_cpu = face_crop.detach().cpu()
            frequency_vec = torch.from_numpy(
                np.stack(
                    [extract_frequency_features(sample) for sample in face_cpu]
                )
            ).to(face_crop.device)

        grayscale = (
            face_crop[:, 0:1] * 0.299
            + face_crop[:, 1:2] * 0.587
            + face_crop[:, 2:3] * 0.114
        )
        forensic_vec = self.forensic_extractor.cnn(
            self.forensic_extractor.srm(grayscale)
        ).flatten(1)
        fused = torch.cat([spatial_vec, frequency_vec, forensic_vec], dim=1)
        return self.fusion_model(fused)


def _as_face_tensor(face_crop: Image.Image | np.ndarray | Tensor) -> tuple[Tensor, np.ndarray]:
    if isinstance(face_crop, Image.Image):
        rgb = np.asarray(face_crop.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    elif isinstance(face_crop, np.ndarray):
        rgb = face_crop
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("face_crop must be an RGB image")
        if rgb.dtype != np.float32 or rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    elif isinstance(face_crop, Tensor):
        tensor = face_crop.detach().float()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("face_crop tensor must contain one image")
            tensor = tensor.squeeze(0)
        if tensor.ndim != 3:
            raise ValueError("face_crop tensor must have three dimensions")
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] != 3:
            raise ValueError("face_crop tensor must have three RGB channels")
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        rgb = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        raise TypeError(f"Unsupported face_crop type: {type(face_crop).__name__}")

    if tuple(tensor.shape[-2:]) != (224, 224):
        raise ValueError("face_crop must be 224x224")
    return tensor.unsqueeze(0), np.clip(rgb, 0.0, 1.0)


def _find_last_spatial_conv(model: SpatialExtractor) -> nn.Conv2d:
    convolutional_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not convolutional_layers:
        raise RuntimeError("Spatial extractor has no convolutional layers")
    name, layer = convolutional_layers[-1]
    if layer.out_channels != 1280:
        raise RuntimeError(
            f"Expected final feature convolution to output 1280 channels; "
            f"{name} outputs {layer.out_channels}"
        )
    print(f"Grad-CAM target layer: {name} ({layer.__class__.__name__}, 1280 channels)")
    return layer


def generate_gradcam(
    face_crop: Image.Image | np.ndarray | Tensor,
    spatial_extractor: SpatialExtractor | None = None,
    forensic_extractor: nn.Module | None = None,
    fusion_model: FusionMLP | None = None,
) -> np.ndarray:
    face_tensor, rgb_image = _as_face_tensor(face_crop)
    spatial_extractor = spatial_extractor or load_spatial_extractor()
    forensic_extractor = forensic_extractor or load_forensic_extractor()
    fusion_model = fusion_model or FusionMLP()
    spatial_extractor.eval()
    forensic_extractor.eval()
    fusion_model.eval()
    wrapper = FusionGradCAMWrapper(
        spatial_extractor, forensic_extractor, fusion_model
    ).eval()
    target_layer = _find_last_spatial_conv(spatial_extractor)

    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=face_tensor,
            targets=[ClassifierOutputTarget(0)],
        )[0]
    return show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)


def generate_forensic_gradcam(
    face_crop: Image.Image | np.ndarray | Tensor,
    spatial_extractor: SpatialExtractor | None = None,
    forensic_extractor: nn.Module | None = None,
    fusion_model: FusionMLP | None = None,
) -> np.ndarray:
    face_tensor, rgb_image = _as_face_tensor(face_crop)
    spatial_extractor = spatial_extractor or load_spatial_extractor()
    forensic_extractor = forensic_extractor or load_forensic_extractor()
    fusion_model = fusion_model or FusionMLP()
    spatial_extractor.eval()
    forensic_extractor.eval()
    fusion_model.eval()
    wrapper = ForensicGradCAMWrapper(
        spatial_extractor, forensic_extractor, fusion_model
    ).eval()
    target_layer = forensic_extractor.cnn[6]
    if not isinstance(target_layer, nn.Conv2d):
        raise RuntimeError("Expected forensic target layer cnn.6 to be Conv2d")
    print(
        "Forensic Grad-CAM target layer: cnn.6 "
        f"({target_layer.__class__.__name__}, {target_layer.out_channels} channels)"
    )

    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=face_tensor,
            targets=[ClassifierOutputTarget(0)],
        )[0]
    return show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
