"""Fine-tune del clasificador de materiales a partir de best.pt (o yolov8n-cls)."""

from pathlib import Path

from ultralytics import YOLO

TRAINED = Path("runs/residuos/weights/best.pt")
BASE = Path("yolov8n-cls.pt")
DATA = Path("dataset_yolo")


if __name__ == "__main__":
    if not (DATA / "train").exists():
        raise SystemExit("Falta dataset_yolo/train. Ejecuta primero: python dividir_dataset.py")

    weights = TRAINED if TRAINED.exists() else BASE
    print(f"Fine-tune desde {weights} sobre {DATA}")
    model = YOLO(str(weights))
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=256,
        batch=32,
        project="runs",
        name="residuos",
        exist_ok=True,
        pretrained=True,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.35,
        fliplr=0.5,
        degrees=12,
        translate=0.08,
        scale=0.15,
        patience=15,
    )
