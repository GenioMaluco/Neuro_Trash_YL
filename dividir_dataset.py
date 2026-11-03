"""
Arma dataset_yolo mezclando TrashNet + capturas de la Kiyo.

Las capturas locales se oversamplean en train si hay pocas, para que el
fine-tune no ignore el dominio real de la cámara.
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

TRASHNET = Path("dataset/Garbage classification")
CAPTURAS = Path("dataset/capturas")
DESTINO = Path("dataset_yolo")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
KIYO_TRAIN_TARGET = 80
random.seed(42)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def collect(root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    if not root.exists():
        return by_class
    for clase in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = list_images(clase)
        if imgs:
            by_class[clase.name] = imgs
    return by_class


def copy_split(images: list[Path], class_name: str, subset: str) -> int:
    out = DESTINO / subset / class_name
    out.mkdir(parents=True, exist_ok=True)
    for src in images:
        dest = out / src.name
        if dest.exists():
            dest = out / f"{src.stem}_{src.parent.name}{src.suffix}"
        shutil.copy2(src, dest)
    return len(images)


def oversample(images: list[Path], target: int) -> list[Path]:
    if not images or len(images) >= target:
        return list(images)
    extras = []
    i = 0
    while len(images) + len(extras) < target:
        extras.append(images[i % len(images)])
        i += 1
    return list(images) + extras


def main() -> None:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    trashnet = collect(TRASHNET)
    capturas = collect(CAPTURAS)
    classes = sorted(set(trashnet) | set(capturas))
    if not classes:
        raise SystemExit("No hay imágenes en dataset/Garbage classification ni en dataset/capturas.")

    counts = defaultdict(lambda: {"train": 0, "val": 0, "kiyo_train": 0, "kiyo_val": 0})
    for class_name in classes:
        public = list(trashnet.get(class_name, []))
        kiyo = list(capturas.get(class_name, []))
        random.shuffle(public)
        random.shuffle(kiyo)

        pub_cut = int(len(public) * 0.8)
        kiyo_cut = int(len(kiyo) * 0.8) if len(kiyo) >= 5 else max(0, len(kiyo) - 1)

        pub_train, pub_val = public[:pub_cut], public[pub_cut:]
        kiyo_train, kiyo_val = kiyo[:kiyo_cut], kiyo[kiyo_cut:]
        kiyo_train_os = oversample(kiyo_train, KIYO_TRAIN_TARGET) if kiyo_train else []

        counts[class_name]["train"] += copy_split(pub_train, class_name, "train")
        counts[class_name]["val"] += copy_split(pub_val, class_name, "val")
        counts[class_name]["kiyo_train"] += copy_split(kiyo_train_os, class_name, "train")
        counts[class_name]["kiyo_val"] += copy_split(kiyo_val, class_name, "val")
        counts[class_name]["train"] += counts[class_name]["kiyo_train"]
        counts[class_name]["val"] += counts[class_name]["kiyo_val"]

    print("dataset_yolo listo (TrashNet + capturas Kiyo):")
    for class_name in classes:
        c = counts[class_name]
        print(
            f"  {class_name:10s}  train={c['train']:4d}  val={c['val']:3d}  "
            f"(kiyo train={c['kiyo_train']}, val={c['kiyo_val']})"
        )
    if not capturas:
        print("Aviso: no hay fotos en dataset/capturas. Captura con captura_imagen.py antes del fine-tune.")


if __name__ == "__main__":
    main()
