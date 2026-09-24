from pathlib import Path
import shutil

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    cached_dataset = Path(
        kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
    )
    DESTINATION.mkdir(parents=True, exist_ok=True)

    for source in cached_dataset.iterdir():
        target = DESTINATION / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    print(f"Copied dataset contents from {cached_dataset} to {DESTINATION}")


if __name__ == "__main__":
    main()
