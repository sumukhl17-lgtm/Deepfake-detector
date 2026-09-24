from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "models" / "weights" / "efficientnet_b0_ffpp"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Xicor9/efficientnet-b0-ffpp-c23",
        local_dir=DESTINATION,
    )
    print(f"Downloaded EfficientNet weights to {DESTINATION}")


if __name__ == "__main__":
    main()
