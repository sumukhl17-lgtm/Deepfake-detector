from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.pipeline import DeepfakePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pipeline = DeepfakePipeline()
    results = {}
    for split_class in ("test/real", "test/fake"):
        image_path = next((PROJECT_ROOT / "data" / "processed" / split_class).glob("*"))
        result = pipeline.run_pipeline(image_path)
        results[split_class] = result
        print(f"\n{split_class}: {result}")
        if "error" not in result:
            confidence = result["confidence"]
            if not isinstance(confidence, float) or not 0.0 <= confidence <= 1.0:
                raise AssertionError(f"Invalid confidence: {confidence!r}")


if __name__ == "__main__":
    main()
