"""
Catálogo Human-in-the-Loop (modelo tipo ER):

  Category 1---N ObjectType 1---N Sample

- Category: CAT-PL, CAT-MT, ... (material / contenedor)
- ObjectType: OBJ-0001 (nombre concreto: botella PET, encendedor, tetra pak)
- Sample: SMP-000001 (recorte + máscara de contorno para fine-tune)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config.settings import (
    CAPTURES_DIR,
    CATEGORIES,
    CATEGORY_BY_ID,
    DATASET_OBJETOS_DIR,
    HITL_CATALOG_PATH,
    HITL_DIR,
    PROJECT_DIR,
    SEG_CLASS_BY_CATEGORY,
    TRAINABLE_CLASSES,
)

REVIEWABLE_CATEGORIES = [name for name in CATEGORIES if name != "No Identificado"]
NEW_OBJECT_TYPE = "Crear tipo nuevo…"
Point = List[int]

DEFAULT_OBJECT_TYPES: Dict[str, List[str]] = {
    "Plástico": [
        "Botella plástica",
        "Botella PET",
        "Encendedor",
        "Cubierto desechable",
        "Envoltorio plástico",
        "Vaso plástico",
        "Bolsa plástica",
    ],
    "Metal": [
        "Cuchara",
        "Tenedor",
        "Lata de aluminio",
        "Papel aluminio",
        "Lata de refresco",
    ],
    "Papel/Cartón": [
        "Caja de cigarros",
        "Tetra Pak",
        "Caja de cartón",
        "Hoja de papel",
        "Vaso de cartón",
        "Caja de pizza",
    ],
    "Orgánico": [
        "Cáscara de fruta",
        "Cáscara de banano",
        "Resto de comida",
        "Cáscara de huevo",
    ],
    "Vidrio": [
        "Botella de vidrio",
        "Frasco de vidrio",
        "Vaso de vidrio",
    ],
    "Desecho General": [
        "Empaque metalizado",
        "Unicel",
        "Colilla",
    ],
}


def attach_entity_ids(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Asigna IDs de categoría y de instancia visibles en el frame (PL-01, MT-02)."""
    for index, det in enumerate(detections, start=1):
        meta = CATEGORIES.get(det.get("category", ""), CATEGORIES["No Identificado"])
        det["category_id"] = meta["category_id"]
        det["category_code"] = meta["code"]
        det["train_class"] = meta.get("train_class")
        det["display_id"] = f"{meta['code']}-{index:02d}"
        det["object_name"] = det.get("subtype") or det.get("category")
        det["hex_color"] = meta["hex"]
    return detections


def _empty_catalog() -> Dict[str, Any]:
    return {
        "categories": [
            {
                "category_id": meta["category_id"],
                "name": name,
                "code": meta["code"],
                "train_class": meta.get("train_class"),
            }
            for name, meta in CATEGORIES.items()
        ],
        "object_types": [],
        "samples": [],
        "next_object_seq": 1,
        "next_sample_seq": 1,
    }


def _seed_default_object_types(catalog: Dict[str, Any]) -> None:
    existing = {
        (item["category_id"], item["name"].lower())
        for item in catalog.get("object_types", [])
    }
    for category, names in DEFAULT_OBJECT_TYPES.items():
        category_id = CATEGORIES[category]["category_id"]
        for name in names:
            key = (category_id, name.lower())
            if key in existing:
                continue
            catalog["object_types"].append(
                {
                    "object_type_id": f"OBJ-{catalog['next_object_seq']:04d}",
                    "category_id": category_id,
                    "name": name,
                    "seeded": True,
                }
            )
            catalog["next_object_seq"] += 1
            existing.add(key)


def load_catalog() -> Dict[str, Any]:
    if HITL_CATALOG_PATH.exists():
        data = json.loads(HITL_CATALOG_PATH.read_text(encoding="utf-8"))
    else:
        data = _empty_catalog()
    data.setdefault("object_types", [])
    data.setdefault("samples", [])
    data.setdefault("next_object_seq", 1)
    data.setdefault("next_sample_seq", 1)
    before = len(data["object_types"])
    _seed_default_object_types(data)
    if len(data["object_types"]) != before or not HITL_CATALOG_PATH.exists():
        save_catalog(data)
    return data


def save_catalog(catalog: Dict[str, Any]) -> None:
    HITL_DIR.mkdir(parents=True, exist_ok=True)
    HITL_CATALOG_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def catalog_stats(catalog: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    data = catalog or load_catalog()
    return {
        "object_types": len(data.get("object_types", [])),
        "samples": len(data.get("samples", [])),
        "masks": sum(1 for sample in data.get("samples", []) if sample.get("mask_path")),
    }


def object_type_options(category: str) -> List[str]:
    catalog = load_catalog()
    category_id = CATEGORIES.get(category, {}).get("category_id")
    names = list(DEFAULT_OBJECT_TYPES.get(category, []))
    for item in catalog.get("object_types", []):
        if item.get("category_id") == category_id and item.get("name") not in names:
            names.append(item["name"])
    return names


def _resolve_object_type(
    catalog: Dict[str, Any],
    category_id: str,
    object_name: str,
) -> str:
    name = object_name.strip() or CATEGORY_BY_ID.get(category_id, "Objeto")
    for obj in catalog["object_types"]:
        if obj["category_id"] == category_id and obj["name"].lower() == name.lower():
            return obj["object_type_id"]
    object_type_id = f"OBJ-{catalog['next_object_seq']:04d}"
    catalog["next_object_seq"] += 1
    catalog["object_types"].append(
        {
            "object_type_id": object_type_id,
            "category_id": category_id,
            "name": name,
        }
    )
    return object_type_id


def infer_train_class(category: str, object_name: str = "") -> Optional[str]:
    meta = CATEGORIES.get(category, CATEGORIES["No Identificado"])
    train_class = meta.get("train_class")
    if category == "Papel/Cartón":
        blob = f"{object_name} {category}".lower()
        if any(token in blob for token in ("cartón", "carton", "cardboard", "caja", "tetra")):
            return "cardboard"
        return "paper"
    return train_class


def contour_to_points(contour: Any) -> List[Point]:
    if contour is None:
        return []
    arr = np.asarray(contour)
    if arr.size < 6:
        return []
    arr = arr.reshape(-1, 2)
    return [[int(x), int(y)] for x, y in arr]


def bbox_as_points(bbox: Sequence[int]) -> List[Point]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def points_to_contour(points: Sequence[Point]) -> Optional[np.ndarray]:
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


def points_to_bbox(points: Sequence[Point]) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def detection_points(det: Dict[str, Any]) -> List[Point]:
    points = contour_to_points(det.get("contour"))
    if len(points) >= 3:
        return points
    return bbox_as_points(det["bbox"])


def densify_points(points: Sequence[Point]) -> List[Point]:
    if len(points) < 2:
        return [list(p) for p in points]
    out: List[Point] = []
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        mid = [int((point[0] + nxt[0]) / 2), int((point[1] + nxt[1]) / 2)]
        out.append([int(point[0]), int(point[1])])
        out.append(mid)
    return out


def draw_points_preview(
    frame: np.ndarray,
    points: Sequence[Point],
    color: Tuple[int, int, int],
    selected: Optional[int] = None,
) -> np.ndarray:
    output = frame.copy()
    if len(points) >= 3:
        contour = points_to_contour(points)
        overlay = output.copy()
        cv2.drawContours(overlay, [contour], -1, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, output, 0.82, 0, output)
        cv2.drawContours(output, [contour], -1, color, thickness=1, lineType=cv2.LINE_AA)
    elif len(points) == 2:
        cv2.line(output, tuple(points[0]), tuple(points[1]), color, 1, cv2.LINE_AA)

    for index, (x, y) in enumerate(points):
        radius = 7 if selected == index else 4
        cv2.circle(output, (int(x), int(y)), radius + 2, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(index + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def crop_from_points(frame: np.ndarray, points: Sequence[Point]) -> Optional[np.ndarray]:
    if len(points) < 3:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = points_to_bbox(points)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    local = np.array([[p[0] - x1, p[1] - y1] for p in points], dtype=np.int32)
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [local], 255)
    gray = np.full_like(crop, 128)
    return np.where(mask[:, :, None] > 0, crop, gray)


def _write_yolo_seg_label(
    image_stem: str,
    points: Sequence[Point],
    image_size: Tuple[int, int],
    class_id: int,
) -> Path:
    h, w = image_size
    coords = []
    for x, y in points:
        coords.append(f"{max(0.0, min(1.0, x / max(w, 1))):.6f}")
        coords.append(f"{max(0.0, min(1.0, y / max(h, 1))):.6f}")
    images_dir = DATASET_OBJETOS_DIR / "images" / "train"
    labels_dir = DATASET_OBJETOS_DIR / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{image_stem}.txt"
    label_path.write_text(f"{class_id} {' '.join(coords)}\n", encoding="utf-8")
    return label_path


def save_reviews(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    source_name: str,
) -> List[Dict[str, Any]]:
    """Persiste recortes, máscaras de contorno y actualiza el catálogo ER."""
    catalog = load_catalog()
    saved: List[Dict[str, Any]] = []
    dets_by_id = {det.get("display_id"): det for det in detections}

    for review in reviews:
        display_id = review["display_id"]
        det = dets_by_id.get(display_id)
        if det is None:
            continue

        category = review["category"]
        if category == "No Identificado":
            continue

        meta = CATEGORIES[category]
        object_name = (review.get("object_name") or det.get("object_name") or category).strip()
        train_class = review.get("train_class") or infer_train_class(category, object_name)
        if train_class not in TRAINABLE_CLASSES:
            continue

        points = review.get("points") or detection_points(det)
        if len(points) < 3:
            continue

        crop = crop_from_points(frame, points)
        if crop is None:
            continue

        dest_dir = CAPTURES_DIR / train_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        next_idx = len(list(dest_dir.glob("*.jpg")))
        crop_path = dest_dir / f"{train_class}_{next_idx:04d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask, [points_to_contour(points)], -1, 255, -1)
        mask_path = dest_dir / f"{train_class}_{next_idx:04d}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        object_type_id = _resolve_object_type(catalog, meta["category_id"], object_name)
        sample_id = f"SMP-{catalog['next_sample_seq']:06d}"
        catalog["next_sample_seq"] += 1

        image_stem = sample_id.lower()
        full_image_path = DATASET_OBJETOS_DIR / "images" / "train" / f"{image_stem}.jpg"
        full_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full_image_path), frame)
        class_id = SEG_CLASS_BY_CATEGORY.get(category)
        label_rel = None
        if class_id is not None:
            label_path = _write_yolo_seg_label(
                image_stem,
                points,
                frame.shape[:2],
                class_id,
            )
            label_rel = str(label_path.relative_to(PROJECT_DIR)).replace("\\", "/")

        sample = {
            "sample_id": sample_id,
            "object_type_id": object_type_id,
            "category_id": meta["category_id"],
            "display_id": display_id,
            "verdict": review.get("verdict", "corrected"),
            "object_name": object_name,
            "predicted_category": det.get("category"),
            "predicted_name": det.get("subtype"),
            "train_class": train_class,
            "source_name": source_name,
            "crop_path": str(crop_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "mask_path": str(mask_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "image_path": str(full_image_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "label_path": label_rel,
            "contour": [[int(x), int(y)] for x, y in points],
            "bbox": list(points_to_bbox(points)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        catalog["samples"].append(sample)
        saved.append(sample)

    if saved:
        save_catalog(catalog)
    return saved


def prepare_dataset() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_DIR / "dividir_dataset.py")],
        cwd=str(PROJECT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )


def start_retrain() -> Path:
    """Lanza el fine-tune en segundo plano y deja el log en dataset/hitl."""
    HITL_DIR.mkdir(parents=True, exist_ok=True)
    log_path = HITL_DIR / "train.log"
    pid_path = HITL_DIR / "train.pid"
    if pid_path.exists():
        raise RuntimeError("Ya hay un reentrenamiento en curso.")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "entrenar_yolo.py")],
            cwd=str(PROJECT_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return log_path


def retrain_running() -> bool:
    pid_path = HITL_DIR / "train.pid"
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return False
    if sys.platform.startswith("win"):
        check = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        alive = str(pid) in check.stdout
    else:
        try:
            import os

            os.kill(pid, 0)
            alive = True
        except PermissionError:
            alive = True
        except OSError:
            alive = False
    if not alive:
        pid_path.unlink(missing_ok=True)
    return alive
