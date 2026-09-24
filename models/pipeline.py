from pathlib import Path
from typing import Any

import torch
from facenet_pytorch import MTCNN
from PIL import Image

from explainability.gradcam import generate_forensic_gradcam, generate_gradcam
from .forensic import extract_forensic_features, load_forensic_extractor
from .frequency import extract_frequency_features
from .fusion import FusionMLP
from .spatial import extract_spatial_features, load_spatial_extractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "weights"


class DeepfakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=224,
            margin=0,
            post_process=False,
            keep_all=False,
            device=self.device,
        )
        self.spatial_extractor = load_spatial_extractor()
        self.forensic_extractor = load_forensic_extractor()
        self.forensic_extractor.load_state_dict(
            torch.load(
                CHECKPOINT_DIR / "forensic_cnn.pth",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.forensic_extractor.eval()
        self.fusion_model = FusionMLP()
        self.fusion_model.load_state_dict(
            torch.load(
                CHECKPOINT_DIR / "fusion_mlp.pth",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.fusion_model.eval()
        self._cpu_mtcnn: MTCNN | None = None

    def _detect_face(self, image: Image.Image) -> torch.Tensor | None:
        try:
            return self.mtcnn(image)
        except RuntimeError as error:
            if self.device.type != "mps" or "MPS" not in str(error):
                raise
            if self._cpu_mtcnn is None:
                self._cpu_mtcnn = MTCNN(
                    image_size=224,
                    margin=0,
                    post_process=False,
                    keep_all=False,
                    device="cpu",
                )
            return self._cpu_mtcnn(image)

    def run_pipeline(self, image_path: str | Path) -> dict[str, Any]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        face_crop = self._detect_face(image)
        if face_crop is None:
            return {"error": "no_face_detected"}

        face_crop = face_crop.detach().cpu()
        face_tensor = face_crop.float() / 255.0
        spatial_vec = extract_spatial_features(self.spatial_extractor, face_tensor)
        frequency_vec = torch.from_numpy(extract_frequency_features(face_crop))
        forensic_vec = extract_forensic_features(self.forensic_extractor, face_tensor)
        label, confidence = self.fusion_model.fuse_and_classify(
            spatial_vec, frequency_vec, forensic_vec
        )
        heatmap = generate_gradcam(
            face_crop,
            spatial_extractor=self.spatial_extractor,
            forensic_extractor=self.forensic_extractor,
            fusion_model=self.fusion_model,
        )
        forensic_heatmap = generate_forensic_gradcam(
            face_crop,
            spatial_extractor=self.spatial_extractor,
            forensic_extractor=self.forensic_extractor,
            fusion_model=self.fusion_model,
        )
        return {
            "label": label,
            "confidence": confidence,
            "face_crop": face_crop,
            "heatmap": heatmap,
            "forensic_heatmap": forensic_heatmap,
        }


def run_pipeline(image_path: str | Path) -> dict[str, Any]:
    return DeepfakePipeline().run_pipeline(image_path)
