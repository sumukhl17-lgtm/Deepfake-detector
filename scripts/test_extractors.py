from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.forensic import extract_forensic_features, load_forensic_extractor
from models.frequency import extract_frequency_features
from models.spatial import extract_spatial_features, load_spatial_extractor


IMAGE_PATH = next((PROJECT_ROOT / "data" / "processed" / "train" / "real").glob("*"))


def run_extractor(name: str, function, expected: int) -> bool:
    try:
        result = function()
        shape = tuple(result.shape)
        passed = shape == (expected,)
        print(f"{name}: shape={shape} ({'PASS' if passed else 'FAIL'})")
        return passed
    except Exception as error:
        print(f"{name}: FAIL - {type(error).__name__}: {error}")
        return False


def main() -> None:
    face = Image.open(IMAGE_PATH).convert("RGB")
    face_tensor = pil_to_tensor(face).float() / 255.0
    results = {
        "spatial": run_extractor(
            "spatial",
            lambda: extract_spatial_features(load_spatial_extractor(), face_tensor),
            1280,
        ),
        "frequency": run_extractor(
            "frequency",
            lambda: torch.from_numpy(extract_frequency_features(face)),
            128,
        ),
        "forensic": run_extractor(
            "forensic",
            lambda: extract_forensic_features(load_forensic_extractor(), face_tensor),
            128,
        ),
    }
    print("\nExtractor summary")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
