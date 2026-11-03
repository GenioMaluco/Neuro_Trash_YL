"""
Motor de Detección y Clasificación Híbrida YOLO (Two-Stage Edge AIoT Engine):
1. Etapa de Localización (Bounding Boxes precisos):
   - Usa YOLOWorld (yolov8s-world.pt) o YOLOv8n (yolov8n.pt) para encontrar las cajas
     delimitadoras exactas de cada objeto en la escena, filtrando personas/muebles.
   - Si el objeto ocupa primer plano y el detector no genera caja, aplica un localizador
     de contorno / Saliency ROI automático para asegurar que el objeto tenga su Frame.
2. Etapa de Refinamiento de Material (Clasificador Especializado TrashNet 92.7%):
   - Extrae la silueta (Canny/GrabCut) dentro de cada caja, recorta por máscara y
     evalúa el material con `runs/residuos/weights/best.pt` (plastic, metal, paper,
     cardboard, glass, trash).
   - Combina inteligentemente la detección semántica (que reconoce frutas/orgánicos,
     baterías, latas, botellas) con la clasificación visual de textura/material del recorte.
   - Si la confianza combinada no supera el umbral de certeza, asigna la categoría
     "No Identificado" (color Violeta del arcoíris) para evitar falsos positivos.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

from config.settings import (
    CATEGORIES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_STABILITY_WINDOW,
    DEFAULT_UNKNOWN_THRESHOLD,
    IGNORED_DETECTOR_CLASSES,
    MODEL_CLASSIFIER_FALLBACK,
    MODEL_CLASSIFIER_TRAINED,
    MODEL_DETECTOR_BASE,
    MODEL_DETECTOR_WORLD,
    PROMPT_TO_CATEGORY,
    TRASHNET_TO_CATEGORY,
    WORLD_WASTE_PROMPTS,
)
from src.hitl import attach_entity_ids
from src.stabilizer import ObjectStabilizer, compute_iou


class NeuroTrashDetector:
    """Detector de residuos multi-objeto con refinamiento de material por recorte y estabilización."""

    def __init__(
        self,
        detector_mode: str = "world_hybrid",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        use_stabilizer: bool = True,
        stability_window: int = DEFAULT_STABILITY_WINDOW,
    ):
        """
        detector_mode:
          - 'world_hybrid': YOLOWorld (prompts de residuos) + Clasificador de material entrenado (best.pt)
          - 'fast_hybrid': YOLOv8n (rápido) + Clasificador de material entrenado (best.pt) + ROI fallback
        """
        self.detector_mode = detector_mode
        self.conf_threshold = conf_threshold
        self.unknown_threshold = unknown_threshold
        self.iou_threshold = iou_threshold
        self.use_stabilizer = use_stabilizer
        self.stabilizer = ObjectStabilizer(window_size=stability_window)

        # 1. Cargar detector de Bounding Boxes
        if detector_mode == "world_hybrid" and MODEL_DETECTOR_WORLD.exists():
            self.detector = YOLOWorld(str(MODEL_DETECTOR_WORLD))
            self.detector.set_classes(WORLD_WASTE_PROMPTS)
            self.detector_name = "YOLOv8s-World + Material-CLS"
        else:
            self.detector = YOLO(str(MODEL_DETECTOR_BASE))
            self.detector_name = "YOLOv8n-Edge + Material-CLS"

        # 2. Cargar clasificador entrenado en residuos (92.7% top-1 accuracy sobre TrashNet)
        cls_path = MODEL_CLASSIFIER_TRAINED if MODEL_CLASSIFIER_TRAINED.exists() else MODEL_CLASSIFIER_FALLBACK
        self.classifier = YOLO(str(cls_path))
        self.classifier_loaded_trained = MODEL_CLASSIFIER_TRAINED.exists()

    def reset_tracker(self) -> None:
        self.stabilizer.reset()

    def _classify_crop_material(self, crop_bgr: np.ndarray) -> Tuple[str, str, float, Dict[str, float]]:
        """Ejecuta el modelo clasificador entrenado (`best.pt`) sobre el recorte exacto del objeto."""
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 16 or crop_bgr.shape[1] < 16:
            return "No Identificado", "Recorte pequeño", 0.0, {}

        res = self.classifier.predict(crop_bgr, imgsz=224, verbose=False)[0]
        if res.probs is None:
            return "No Identificado", "Sin probabilidades", 0.0, {}

        top1_idx = int(res.probs.top1)
        top1_conf = float(res.probs.top1conf)
        raw_name = res.names[top1_idx].lower()

        probs_dict = {}
        data_list = res.probs.data.tolist()
        for idx, score in enumerate(data_list):
            probs_dict[res.names[idx].lower()] = float(score)

        cat, subtype = TRASHNET_TO_CATEGORY.get(raw_name, ("No Identificado", raw_name))
        return cat, subtype, top1_conf, probs_dict

    @staticmethod
    def _edge_foreground_mask(bgr: np.ndarray) -> np.ndarray:
        """Máscara de primer plano por Canny + morfología (misma idea que el ROI de respaldo)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.dilate(closed, kernel, iterations=1)

    @staticmethod
    def _largest_filled_contour(
        mask: np.ndarray,
        min_area: float,
        max_area: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(best))
        if area < min_area:
            return None
        if max_area is not None and area > max_area:
            return None
        return best

    def _find_foreground_object_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Localizador de respaldo cuando el objeto está en primer plano pero el detector genérico
        no generó caja: segmenta el objeto principal respecto al fondo para crear un Frame preciso.
        """
        h, w = frame.shape[:2]
        closed = self._edge_foreground_mask(frame)
        min_area = (h * w) * 0.035
        max_area = (h * w) * 0.92
        best_cnt = None
        best_area = 0.0

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = float(cw * ch)
            if min_area <= area <= max_area and area > best_area:
                best_area = area
                best_cnt = (x, y, cw, ch)

        if best_cnt is None:
            return None

        x, y, cw, ch = best_cnt
        pad_x = int(cw * 0.04)
        pad_y = int(ch * 0.04)
        return (
            max(4, x - pad_x),
            max(4, y - pad_y),
            min(w - 4, x + cw + pad_x),
            min(h - 4, y + ch + pad_y),
        )

    def _grabcut_mask(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Segmenta el objeto con GrabCut en un recorte reducido (1 iteración)."""
        ch, cw = crop_bgr.shape[:2]
        if ch < 40 or cw < 40:
            return None
        max_side = 160
        scale = min(1.0, max_side / float(max(ch, cw)))
        work = crop_bgr
        if scale < 1.0:
            work = cv2.resize(
                crop_bgr,
                (max(40, int(cw * scale)), max(40, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )
        wh, ww = work.shape[:2]
        inset_x = max(2, int(ww * 0.06))
        inset_y = max(2, int(wh * 0.06))
        rw = ww - 2 * inset_x
        rh = wh - 2 * inset_y
        if rw < 12 or rh < 12:
            return None
        gc_mask = np.zeros((wh, ww), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                work,
                gc_mask,
                (inset_x, inset_y, rw, rh),
                bgd,
                fgd,
                1,
                cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            return None
        binary = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        if cv2.countNonZero(binary) < (wh * ww) * 0.08:
            return None
        if binary.shape[:2] != (ch, cw):
            binary = cv2.resize(binary, (cw, ch), interpolation=cv2.INTER_NEAREST)
        return binary

    def _extract_object_silhouette(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extrae silueta (contorno + máscara) del objeto dentro de la caja.
        Primero Canny+morfología; si queda débil, refina con GrabCut.
        El contorno va en coordenadas del frame completo.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None, None

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        crop_area = float(ch * cw)
        min_area = crop_area * 0.06
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask_local = None
        gc_mask = self._grabcut_mask(crop)
        if gc_mask is not None:
            gc_cover = cv2.countNonZero(gc_mask) / crop_area
            if 0.08 <= gc_cover <= 0.88:
                mask_local = gc_mask

        if mask_local is None:
            edge_mask = self._edge_foreground_mask(crop)
            edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contour_local = self._largest_filled_contour(edge_mask, min_area)
            if contour_local is None:
                return None, None
            mask_local = np.zeros((ch, cw), np.uint8)
            cv2.drawContours(mask_local, [contour_local], -1, 255, -1)

        mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_OPEN, kernel, iterations=1)
        # Corta brazos delgados (picos de Canny) conservando el núcleo del objeto
        dist = cv2.distanceTransform(mask_local, cv2.DIST_L2, 5)
        max_d = float(dist.max())
        if max_d >= 6.0:
            cut = max(2.0, max_d * 0.16)
            _, cores = cv2.threshold(dist, cut, 255, cv2.THRESH_BINARY)
            cores = cv2.dilate(cores.astype(np.uint8), kernel, iterations=2)
            if cv2.countNonZero(cores) >= min_area:
                mask_local = cores

        blur_k = 7 if min(ch, cw) >= 80 else 5
        blurred = cv2.GaussianBlur(mask_local, (blur_k, blur_k), 0)
        _, mask_local = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)

        contour_local = self._largest_filled_contour(mask_local, min_area)
        if contour_local is None:
            return None, None

        area = float(cv2.contourArea(contour_local))
        peri = cv2.arcLength(contour_local, True)
        compactness = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        if compactness < 0.10:
            return None, None

        epsilon = max(0.6, 0.0015 * peri)
        approx = cv2.approxPolyDP(contour_local, epsilon, True)
        if 8 <= len(approx) <= len(contour_local):
            contour_local = approx

        contour = contour_local.astype(np.int32).copy()
        contour[:, 0, 0] += x1
        contour[:, 0, 1] += y1

        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y1:y2, x1:x2] = mask_local
        return contour, full_mask

    def _masked_material_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Recorta por máscara/contorno: menos fondo de mesa, mejor señal de material."""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return crop
        if mask is None:
            return crop

        local = mask[y1:y2, x1:x2]
        if local.size == 0 or cv2.countNonZero(local) < 32:
            return crop

        ys, xs = np.where(local > 0)
        pad = 2
        mx1 = max(0, int(xs.min()) - pad)
        mx2 = min(local.shape[1], int(xs.max()) + 1 + pad)
        my1 = max(0, int(ys.min()) - pad)
        my2 = min(local.shape[0], int(ys.max()) + 1 + pad)
        tight = crop[my1:my2, mx1:mx2]
        tight_mask = local[my1:my2, mx1:mx2]
        if tight.size == 0:
            return crop

        k = 31 if min(tight.shape[:2]) >= 48 else 11
        if k % 2 == 0:
            k += 1
        soft_bg = cv2.GaussianBlur(tight, (k, k), 0)
        return np.where(tight_mask[:, :, None] > 0, tight, soft_bg)

    def _fuse_predictions(
        self,
        det_label: str,
        det_conf: float,
        cls_cat: str,
        cls_subtype: str,
        cls_conf: float,
        cls_probs: Dict[str, float],
    ) -> Tuple[str, str, float]:
        """
        Fusión experta entre el Detector de Objetos (YOLOWorld/YOLOv8) y el Clasificador de Material (best.pt):
        - Corrige el problema clásico donde una botella plástica transparente frente a la cámara se confundía con vidrio:
          combina la evidencia de botella plástica/PET con las probabilidades de material.
        - Preserva la categoría 'Orgánico' cuando el detector identifica frutas, cáscaras, vegetales o restos de comida
          (ya que TrashNet original no tenía carpeta 'organic').
        - Marca como 'No Identificado' cuando la confianza final está por debajo de `self.unknown_threshold`.
        """
        det_cat, det_subtype = PROMPT_TO_CATEGORY.get(
            det_label, ("No Identificado", det_label.capitalize())
        )

        # Caso 1: El detector identifica claramente un residuo Orgánico (fruta, cáscara, comida, planta)
        if det_cat == "Orgánico" and det_conf >= 0.22:
            fused_conf = min(0.99, max(det_conf * 1.15, (det_conf * 0.7 + cls_conf * 0.3)))
            if fused_conf < self.unknown_threshold:
                return "No Identificado", f"Posible {det_subtype}", fused_conf
            return "Orgánico", det_subtype, fused_conf

        # Caso 1b: El clasificador ya conoce la clase organic (tras reentrenar con capturas Kiyo)
        if cls_cat == "Orgánico" and cls_conf >= 0.45:
            fused_conf = min(0.99, 0.70 * cls_conf + 0.30 * max(det_conf, 0.35))
            if fused_conf >= self.unknown_threshold:
                return "Orgánico", cls_subtype, fused_conf

        # Caso 2: Baterías, electrónicos o desechos especiales detectados explícitamente
        if det_label in {
            "AA battery",
            "battery",
            "computer mouse",
            "cell phone",
            "snack wrapper",
            "candy wrapper",
            "chip bag",
            "styrofoam",
        } and det_conf >= 0.22:
            fused_conf = min(0.99, max(det_conf, 0.75))
            return "Desecho General", det_subtype, fused_conf

        # Caso 3: Acuerdos entre Detector y Clasificador de Material (Máxima precisión)
        if det_cat == cls_cat and det_cat != "No Identificado":
            fused_conf = min(0.99, 0.45 * det_conf + 0.55 * cls_conf + 0.12)
            return det_cat, det_subtype, fused_conf

        # Caso 4: Corrección inteligente de Botellas Plásticas vs Vidrio (Problema visto en captura_1.jpg)
        # Si el detector ve una botella plástica/agua pero el clasificador dudó entre glass y plastic por el brillo:
        plastic_prob = cls_probs.get("plastic", 0.0)
        glass_prob = cls_probs.get("glass", 0.0)
        plastic_bottle_labels = {
            "plastic bottle",
            "water bottle",
            "soda bottle",
            "plastic cup",
            "bottle",
            "PET bottle",
            "shampoo bottle",
            "plastic wrapper",
            "food wrapper",
        }
        if det_label in plastic_bottle_labels and cls_cat == "Vidrio":
            if det_label != "bottle" and (det_conf >= 0.28 or plastic_prob >= 0.12):
                fused_conf = max(det_conf, min(0.94, cls_conf * 0.85 + plastic_prob))
                return "Plástico", "Botella PET / Plástico", fused_conf

        # Caso 5: Si el clasificador de materiales entrenado tiene alta seguridad (>= 0.68) en el recorte
        if cls_conf >= 0.68 and cls_cat != "No Identificado":
            subtype = det_subtype if det_cat == cls_cat else cls_subtype
            fused_conf = 0.75 * cls_conf + 0.25 * max(det_conf, 0.4)
            if fused_conf >= self.unknown_threshold:
                return cls_cat, subtype, fused_conf

        # Caso 6: Si el detector tiene buena confianza aunque el clasificador esté inseguro
        if det_conf >= 0.45 and det_cat != "No Identificado":
            return det_cat, det_subtype, det_conf

        # Caso 7: Promedio ponderado general
        best_cat = cls_cat if cls_conf >= det_conf else det_cat
        best_sub = cls_subtype if cls_conf >= det_conf else det_subtype
        best_conf = max(cls_conf * 0.85, det_conf)

        if best_conf < self.unknown_threshold or best_cat == "No Identificado":
            return "No Identificado", "Objeto no identificado", best_conf

        return best_cat, best_sub, best_conf

    def analyze_frame(
        self,
        frame: np.ndarray,
        enable_fallback_roi: bool = True,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Analiza un fotograma BGR y devuelve:
          - Lista de objetos detectados con su bounding box, categoría de las 7 del arcoíris,
            subtipo, confianza y contenedor destino.
          - Latencia total de inferencia en milisegundos (ms).
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # 1. Ejecutar detector de objetos YOLO para obtener los Bounding Boxes
        det_results = self.detector.predict(
            frame,
            imgsz=640,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        raw_candidates: List[Dict[str, Any]] = []

        if det_results.boxes is not None and len(det_results.boxes) > 0:
            for box in det_results.boxes:
                cls_id = int(box.cls[0])
                det_label = det_results.names[cls_id]
                if det_label in IGNORED_DETECTOR_CLASSES:
                    continue

                det_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) < 18 or (y2 - y1) < 18:
                    continue

                raw_candidates.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "det_label": det_label,
                        "det_conf": det_conf,
                    }
                )

        # Ordenar por confianza descendente y filtrar cajas duplicadas (NMS agnóstico de clase)
        raw_candidates.sort(key=lambda item: item["det_conf"], reverse=True)
        filtered_candidates: List[Dict[str, Any]] = []
        for cand in raw_candidates:
            if any(compute_iou(cand["bbox"], kept["bbox"]) > 0.65 for kept in filtered_candidates):
                continue
            filtered_candidates.append(cand)
            if len(filtered_candidates) >= 8:  # Máximo 8 objetos simultáneos para mantener baja latencia Edge
                break

        # 2. Si no hubo cajas del detector pero hay un objeto claro en primer plano, localizar su contorno
        if not filtered_candidates and enable_fallback_roi:
            roi_box = self._find_foreground_object_box(frame)
            if roi_box is not None:
                filtered_candidates.append(
                    {
                        "bbox": roi_box,
                        "det_label": "unknown",
                        "det_conf": 0.30,
                    }
                )

        # 3. Silueta dentro de cada caja + recorte por máscara para clasificar material
        detections: List[Dict[str, Any]] = []
        for cand in filtered_candidates:
            x1, y1, x2, y2 = cand["bbox"]
            contour, mask = self._extract_object_silhouette(frame, (x1, y1, x2, y2))
            crop = self._masked_material_crop(frame, (x1, y1, x2, y2), mask)
            cls_cat, cls_subtype, cls_conf, cls_probs = self._classify_crop_material(crop)

            final_cat, final_subtype, final_conf = self._fuse_predictions(
                det_label=cand["det_label"],
                det_conf=cand["det_conf"],
                cls_cat=cls_cat,
                cls_subtype=cls_subtype,
                cls_conf=cls_conf,
                cls_probs=cls_probs,
            )

            cat_meta = CATEGORIES.get(final_cat, CATEGORIES["No Identificado"])
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "contour": contour,
                    "category": final_cat,
                    "subtype": final_subtype,
                    "confidence": round(float(final_conf), 4),
                    "rainbow_color_name": cat_meta["rainbow_name"],
                    "hex_color": cat_meta["hex"],
                    "container": cat_meta["container"],
                    "servo_angle": cat_meta["servo_angle"],
                    "recyclable": cat_meta["recyclable"],
                }
            )

        # 4. Estabilizar temporalmente los Frames y categorías si está activo
        if self.use_stabilizer:
            detections = self.stabilizer.update(detections)
            # Re-sincronizar metadatos del arcoíris tras la votación temporal
            for det in detections:
                cat_meta = CATEGORIES.get(det["category"], CATEGORIES["No Identificado"])
                det["rainbow_color_name"] = cat_meta["rainbow_name"]
                det["hex_color"] = cat_meta["hex"]
                det["container"] = cat_meta["container"]
                det["servo_angle"] = cat_meta["servo_angle"]
                det["recyclable"] = cat_meta["recyclable"]

        attach_entity_ids(detections)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return detections, latency_ms
