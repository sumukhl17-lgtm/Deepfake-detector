from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from explainability.gradcam import generate_gradcam


def main() -> None:
    image_path = next((PROJECT_ROOT / "data" / "processed" / "test" / "real").glob("*"))
    overlay = generate_gradcam(Image.open(image_path).convert("RGB"))
    output_path = PROJECT_ROOT / "data" / "processed" / "gradcam_test_output.png"
    Image.fromarray(overlay).save(output_path)
    print(f"Saved Grad-CAM overlay: {output_path}")
    print(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}")


if __name__ == "__main__":
    main()
