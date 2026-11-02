"""
Estabilizador temporal de objetos (Multi-Object IoU Tracker + Votación por Ventana):
Suaviza las coordenadas de los frames (bounding boxes) y estabiliza la categoría
predicha a lo largo de N fotogramas consecutivos para eliminar parpadeos y falsos positivos.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter_area / float(area_a + area_b - inter_area)


@dataclass
class TrackedObject:
    track_id: int
    bbox: Tuple[int, int, int, int]
    categories: deque = field(default_factory=lambda: deque(maxlen=7))
    subtypes: deque = field(default_factory=lambda: deque(maxlen=7))
    confidences: deque = field(default_factory=lambda: deque(maxlen=7))
    missed_frames: int = 0
    hits: int = 0

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        category: str,
        subtype: str,
        confidence: float,
        smooth_alpha: float = 0.65,
    ) -> None:
        # Suavizado exponencial de las coordenadas para que el frame no tiemble
        x1 = int(smooth_alpha * bbox[0] + (1.0 - smooth_alpha) * self.bbox[0])
        y1 = int(smooth_alpha * bbox[1] + (1.0 - smooth_alpha) * self.bbox[1])
        x2 = int(smooth_alpha * bbox[2] + (1.0 - smooth_alpha) * self.bbox[2])
        y2 = int(smooth_alpha * bbox[3] + (1.0 - smooth_alpha) * self.bbox[3])
        self.bbox = (x1, y1, x2, y2)
        self.categories.append(category)
        self.subtypes.append(subtype)
        self.confidences.append(confidence)
        self.missed_frames = 0
        self.hits += 1

    @property
    def stable_category(self) -> str:
        if not self.categories:
            return "No Identificado"
        return Counter(self.categories).most_common(1)[0][0]

    @property
    def stable_subtype(self) -> str:
        if not self.subtypes:
            return "Objeto en análisis"
        target_cat = self.stable_category
        matching = [
            sub for cat, sub in zip(self.categories, self.subtypes) if cat == target_cat
        ]
        if matching:
            return Counter(matching).most_common(1)[0][0]
        return Counter(self.subtypes).most_common(1)[0][0]

    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class ObjectStabilizer:
    """Mantiene el seguimiento de objetos entre fotogramas para aumentar la precisión."""

    def __init__(self, window_size: int = 5, iou_match_threshold: float = 0.3, max_missed: int = 4):
        self.window_size = max(1, window_size)
        self.iou_match_threshold = iou_match_threshold
        self.max_missed = max_missed
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Recibe detecciones crudas [{'bbox': (x1,y1,x2,y2), 'category': ..., 'subtype': ..., 'confidence': ...}]
        y devuelve las detecciones con bbox suavizado, categoría estabilizada y track_id.
        """
        used_tracks = set()
        updated_detections = []

        for det in detections:
            bbox = det["bbox"]
            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                iou = compute_iou(bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_match_threshold:
                track = self.tracks[best_track_id]
                track.update(
                    bbox=bbox,
                    category=det["category"],
                    subtype=det["subtype"],
                    confidence=det["confidence"],
                )
                used_tracks.add(best_track_id)
            else:
                track_id = self._next_id
                self._next_id += 1
                track = TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    categories=deque(maxlen=self.window_size),
                    subtypes=deque(maxlen=self.window_size),
                    confidences=deque(maxlen=self.window_size),
                )
                track.update(
                    bbox=bbox,
                    category=det["category"],
                    subtype=det["subtype"],
                    confidence=det["confidence"],
                    smooth_alpha=1.0,
                )
                self.tracks[track_id] = track
                used_tracks.add(track_id)

            enriched = dict(det)
            enriched["track_id"] = track.track_id
            enriched["bbox"] = track.bbox
            enriched["category"] = track.stable_category
            enriched["subtype"] = track.stable_subtype
            enriched["confidence"] = round(track.avg_confidence, 4)
            enriched["is_confirmed"] = track.hits >= min(2, self.window_size)
            updated_detections.append(enriched)

        # Limpiar tracks antiguos que ya salieron de escena
        stale_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in used_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]

        return updated_detections
