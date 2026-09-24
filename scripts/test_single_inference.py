from pathlib import Path
import sys

import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.forensic import extract_forensic_features, load_forensic_extractor
from models.fusion import FusionMLP, fuse_and_classify


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir = PROJECT_ROOT / "models" / "weights"
    cache = torch.load(
        PROJECT_ROOT / "data" / "processed" / "features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    real_index = next(
        index for index, label in enumerate(cache["labels"]) if int(label) == 0
    )
    filename = Path(cache["filenames"][real_index])
    image_path = PROJECT_ROOT / "data" / "processed" / filename
    with Image.open(image_path) as image:
        face_tensor = pil_to_tensor(image.convert("RGB")).float() / 255.0

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

    forensic_features = extract_forensic_features(
        forensic, face_tensor.to(device)
    )
    label, confidence = fuse_and_classify(
        cache["spatial"][real_index].to(device),
        cache["frequency"][real_index].to(device),
        forensic_features,
        model=fusion,
    )
    result = (label, confidence)
    if (
        not isinstance(label, str)
        or label not in {"REAL", "FAKE"}
        or not isinstance(confidence, float)
        or not 0.0 <= confidence <= 1.0
    ):
        raise AssertionError(f"Invalid inference result: {result!r}")
    print(f"Test feature: {filename}")
    print(f"Inference result: {result}")


if __name__ == "__main__":
    main()
