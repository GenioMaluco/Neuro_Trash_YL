"""
Catálogo Human-in-the-Loop (modelo tipo ER):

  Category 1---N ObjectType 1---N Sample

- Category: CAT-PL, CAT-MT, ... (material / contenedor)
- ObjectType: OBJ-0001 (nombre concreto: botella PET, cáscara, lata)
- Sample: SMP-000001 (recorte etiquetado para fine-tune)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    CAPTURES_DIR,
    CATEGORIES,
    CATEGORY_BY_ID,
    HITL_CATALOG_PATH,
    HITL_DIR,
    PROJECT_DIR,
    TRAINABLE_CLASSES,
)

REVIEWABLE_CATEGORIES = [name for name in CATEGORIES if name != "No Identificado"]


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


def load_catalog() -> Dict[str, Any]:
    if not HITL_CATALOG_PATH.exists():
        return _empty_catalog()
    data = json.loads(HITL_CATALOG_PATH.read_text(encoding="utf-8"))
    data.setdefault("object_types", [])
    data.setdefault("samples", [])
    data.setdefault("next_object_seq", 1)
    data.setdefault("next_sample_seq", 1)
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
    }


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
        if any(token in blob for token in ("cartón", "carton", "cardboard", "caja")):
            return "cardboard"
        return "paper"
    return train_class


def crop_detection(frame: np.ndarray, det: Dict[str, Any]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    return frame[y1:y2, x1:x2].copy()


def save_reviews(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    source_name: str,
) -> List[Dict[str, Any]]:
    """Persiste recortes etiquetados y actualiza el catálogo ER."""
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

        crop = crop_detection(frame, det)
        if crop is None:
            continue

        dest_dir = CAPTURES_DIR / train_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        next_idx = len(list(dest_dir.glob("*.jpg")))
        crop_path = dest_dir / f"{train_class}_{next_idx:04d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        object_type_id = _resolve_object_type(catalog, meta["category_id"], object_name)
        sample_id = f"SMP-{catalog['next_sample_seq']:06d}"
        catalog["next_sample_seq"] += 1
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
            "bbox": [int(v) for v in det["bbox"]],
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
