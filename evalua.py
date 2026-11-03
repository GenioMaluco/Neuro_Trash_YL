"""Evalúa el clasificador actual (TrashNet val y, si existen, capturas Kiyo)."""

from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("runs/residuos/weights/best.pt")
YOLO_VAL = Path("dataset_yolo")
KIYO = Path("dataset/capturas")


if __name__ == "__main__":
    if not WEIGHTS.exists():
        raise SystemExit(f"No existe {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    if (YOLO_VAL / "val").exists():
        metrics = model.val(data=str(YOLO_VAL), imgsz=256, verbose=False)
        print(f"dataset_yolo val  top1={metrics.top1:.4f}  top5={metrics.top5:.4f}")
    else:
        print("No hay dataset_yolo/val. Ejecuta dividir_dataset.py")

    if KIYO.exists() and any(KIYO.rglob("*.jpg")):
        print(f"Hay capturas Kiyo en {KIYO}. Tras el fine-tune, vuelve a correr este script.")
    else:
        print("Aún no hay capturas Kiyo. Usa: python captura_imagen.py --clase plastic")
