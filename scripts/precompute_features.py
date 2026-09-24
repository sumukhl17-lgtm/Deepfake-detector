from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.frequency import extract_frequency_features
from models.spatial import extract_spatial_features, load_spatial_extractor


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 64


def image_paths(split: str, class_name: str) -> list[Path]:
    directory = PROCESSED_ROOT / split / class_name
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    spatial_model = load_spatial_extractor().to(device)
    all_paths = {
        split: [
            (path, 0 if class_name == "real" else 1)
            for class_name in ("real", "fake")
            for path in image_paths(split, class_name)
        ]
        for split in ("train", "val", "test")
    }

    for split, entries in all_paths.items():
        spatial_features: list[torch.Tensor] = []
        frequency_features: list[torch.Tensor] = []
        labels: list[int] = []
        filenames: list[str] = []

        for start in tqdm(range(0, len(entries), BATCH_SIZE), desc=f"{split} ({device})"):
            batch_entries = entries[start : start + BATCH_SIZE]
            tensors = []
            batch_images: list[Image.Image] = []
            for path, label in batch_entries:
                image = Image.open(path).convert("RGB")
                batch_images.append(image)
                tensors.append(pil_to_tensor(image).float() / 255.0)

            batch_tensor = torch.stack(tensors).to(device)
            spatial_batch = extract_spatial_features(spatial_model, batch_tensor)
            spatial_features.append(spatial_batch.cpu())

            for (path, label), image in zip(batch_entries, batch_images):
                frequency_features.append(
                    torch.from_numpy(extract_frequency_features(image))
                )
                labels.append(label)
                filenames.append(str(path.relative_to(PROCESSED_ROOT)))
                image.close()

        output = {
            "spatial": torch.cat(spatial_features),
            "frequency": torch.stack(frequency_features),
            "labels": torch.tensor(labels, dtype=torch.long),
            "filenames": filenames,
        }
        output_path = PROCESSED_ROOT / f"features_{split}.pt"
        torch.save(output, output_path)
        print(
            f"{output_path}: spatial={tuple(output['spatial'].shape)}, "
            f"frequency={tuple(output['frequency'].shape)}, "
            f"labels={tuple(output['labels'].shape)}, "
            f"filenames={len(output['filenames'])}"
        )


if __name__ == "__main__":
    main()
