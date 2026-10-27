

from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = Path("models/best.pt")


def load_defect_model():
    """Load the trained YOLO model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


def detect_defects(image_path, confidence=0.5):
    """Detect product defects in an image."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_defect_model()

    results = model.predict(
        source=str(image_path),
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Save image with detection boxes
    output_dir = Path("output/detected_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_detected.jpg"

    annotated_image = result.plot()
    cv2.imwrite(str(output_path), annotated_image)

    # Extract defect information
    defects = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())

        defects.append({
            "defect_type": result.names[class_id],
            "confidence": round(
                float(box.conf[0].item()) * 100, 2
            )
        })

    return {
        "total_defects": len(defects),
        "status": (
            "Defective" if defects else "No Defects Detected"
        ),
        "defects": defects,
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    image_path = input("Enter product image path: ")

    try:
        report = detect_defects(image_path)

        print("\n--- Inspection Report ---")
        print("Status:", report["status"])
        print("Total Defects:", report["total_defects"])

        for defect in report["defects"]:
            print(
                defect["defect_type"],
                "- Confidence:",
                str(defect["confidence"]) + "%"
            )

        print("Detected image:", report["output_path"])

    except Exception as error:
        print("Error:", error)
