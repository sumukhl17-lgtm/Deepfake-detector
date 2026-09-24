from pathlib import Path
import random

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "real_vs_fake"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
SKIPPED_LOG = PROCESSED_ROOT / "skipped_no_face.txt"

SAMPLE_CAPS = {
    "train": 8_000,
    "val": 1_500,
    "test": 1_500,
}
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split_class_directories() -> dict[str, dict[str, Path]]:
    split_directories: dict[str, dict[str, Path]] = {}
    split_aliases = {"train": "train", "valid": "val", "validation": "val", "test": "test"}

    for directory in (path for path in RAW_ROOT.rglob("*") if path.is_dir()):
        split_name = split_aliases.get(directory.name.casefold())
        if split_name is None:
            continue
        classes: dict[str, Path] = {}
        for child in directory.iterdir():
            if not child.is_dir():
                continue
            class_name = child.name.casefold()
            if "real" in class_name and "real" not in classes:
                classes["real"] = child
            elif "fake" in class_name and "fake" not in classes:
                classes["fake"] = child
        if len(classes) == 2:
            split_directories[split_name] = classes

    missing = set(SAMPLE_CAPS) - set(split_directories)
    if missing:
        raise FileNotFoundError(f"Could not discover splits: {', '.join(sorted(missing))}")
    return split_directories


def sample_images(directory: Path, cap: int, rng: random.Random) -> list[Path]:
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    if len(images) > cap:
        return sorted(rng.sample(images, cap))
    return images


def main() -> None:
    directories = find_split_class_directories()
    rng = random.Random(RANDOM_SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mtcnn = create_mtcnn(device)
    cpu_mtcnn: MTCNN | None = None
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    summary: dict[tuple[str, str], tuple[int, int]] = {}

    for split, cap in SAMPLE_CAPS.items():
        for class_name in ("real", "fake"):
            images = sample_images(directories[split][class_name], cap, rng)
            destination = PROCESSED_ROOT / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            processed = 0

            for source in tqdm(images, desc=f"{split}/{class_name} ({device})"):
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    try:
                        crop = mtcnn(image)
                    except RuntimeError as error:
                        if device.type != "mps" or "MPS" not in str(error):
                            raise
                        if cpu_mtcnn is None:
                            cpu_mtcnn = create_mtcnn(torch.device("cpu"))
                        crop = cpu_mtcnn(image)
                if crop is None:
                    skipped.append(f"{split}/{class_name}/{source.name}")
                    continue
                crop_image = Image.fromarray(crop.permute(1, 2, 0).byte().cpu().numpy())
                crop_image.save(destination / source.name)
                processed += 1

            summary[(split, class_name)] = (processed, len(images) - processed)

    SKIPPED_LOG.write_text("\n".join(skipped) + ("\n" if skipped else ""))
    print("\nSummary")
    print(f"{'split':<8}{'class':<8}{'processed':>12}{'skipped':>10}")
    for split in SAMPLE_CAPS:
        for class_name in ("real", "fake"):
            processed, skipped_count = summary[(split, class_name)]
            print(f"{split:<8}{class_name:<8}{processed:>12}{skipped_count:>10}")


def create_mtcnn(device: torch.device) -> MTCNN:
    return MTCNN(
        image_size=224,
        margin=0,
        post_process=False,
        keep_all=False,
        device=device,
    )


if __name__ == "__main__":
    main()
