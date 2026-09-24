from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class CachedFeatureDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        cache = torch.load(
            PROCESSED_ROOT / f"features_{split}.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.spatial = cache["spatial"].float()
        self.frequency = cache["frequency"].float()
        self.labels = cache["labels"].long()
        self.filenames = cache["filenames"]
        self.split = split

        if not (
            len(self.spatial)
            == len(self.frequency)
            == len(self.labels)
            == len(self.filenames)
        ):
            raise ValueError(f"Inconsistent cached feature lengths for {split}")

    def __len__(self) -> int:
        return len(self.labels)

    def _crop_path(self, index: int) -> Path:
        label = int(self.labels[index])
        class_name = "fake" if label == 1 else "real"
        filename = Path(self.filenames[index])
        if filename.parts[:2] == (self.split, class_name):
            filename = Path(*filename.parts[2:])
        return PROCESSED_ROOT / self.split / class_name / filename

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        crop_path = self._crop_path(index)
        with Image.open(crop_path) as image:
            raw_face_crop = pil_to_tensor(image.convert("RGB")).float() / 255.0
        return (
            self.spatial[index],
            self.frequency[index],
            raw_face_crop,
            self.labels[index].float(),
        )
