"""
Renderizador visual de Frames con la paleta de colores del Arcoíris:
Dibuja un contorno de 1 px que sigue la silueta del objeto (o una caja delgada
si el contorno falla), etiqueta compacta y panel HUD con telemetría Edge/IoT.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import CATEGORIES, TARGET_LATENCY_MS


CONTOUR_FILL_ALPHA = 0.16
NEON_GLOW_ALPHA = 0.32


def get_text_color_for_bg(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Calcula la luminancia del color de fondo para elegir texto blanco o negro."""
    b, g, r = bgr
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return (15, 15, 15) if luminance > 0.58 else (255, 255, 255)


def _valid_contour(contour: Any) -> Optional[np.ndarray]:
    if contour is None:
        return None
    arr = np.asarray(contour)
    if arr.size < 6:
        return None
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr.reshape(-1, 1, 2)
    if arr.ndim != 3 or arr.shape[1] != 1 or arr.shape[2] != 2:
        return None
    if len(arr) < 3:
        return None
    return arr.astype(np.int32)


def _draw_compact_label(
    image: np.ndarray,
    anchor_x: int,
    anchor_y: int,
    label: str,
    color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
) -> None:
    """Etiqueta pequeña anclada al punto más alto, sin placa grande."""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    font_thick = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    pad_x, pad_y = 4, 2
    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2

    lx1 = int(np.clip(anchor_x - box_w // 2, 2, max(2, w - box_w - 2)))
    ly2 = anchor_y - 3
    ly1 = ly2 - box_h
    if ly1 < 2:
        ly1 = min(h - box_h - 2, anchor_y + 3)
        ly2 = ly1 + box_h
    lx2 = min(w - 2, lx1 + box_w)

    cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (lx1 + pad_x, ly2 - pad_y - max(0, baseline // 2)),
        font,
        font_scale,
        text_color,
        font_thick,
        cv2.LINE_AA,
    )


def draw_rainbow_frame(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    category: str,
    subtype: str,  # conservado por compatibilidad con draw_hud
    confidence: float,
    track_id: int = 0,
    contour: Any = None,
    display_id: str = "",
) -> None:
    """Dibuja silueta neón de 1 px (o caja delgada) con ID de categoría."""
    cat_info = CATEGORIES.get(category, CATEGORIES["No Identificado"])
    color = cat_info["bgr"]
    text_color = get_text_color_for_bg(color)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(1, x1), max(1, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    shape = _valid_contour(contour)
    if shape is not None:
        glow = image.copy()
        cv2.drawContours(glow, [shape], -1, color, thickness=5, lineType=cv2.LINE_AA)
        cv2.addWeighted(glow, NEON_GLOW_ALPHA, image, 1.0 - NEON_GLOW_ALPHA, 0, image)
        overlay = image.copy()
        cv2.drawContours(overlay, [shape], -1, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, CONTOUR_FILL_ALPHA, image, 1.0 - CONTOUR_FILL_ALPHA, 0, image)
        cv2.drawContours(image, [shape], -1, color, thickness=1, lineType=cv2.LINE_AA)
        pts = shape.reshape(-1, 2)
        top = pts[int(np.argmin(pts[:, 1]))]
        anchor_x, anchor_y = int(top[0]), int(top[1])
    else:
        glow = image.copy()
        cv2.rectangle(glow, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
        cv2.addWeighted(glow, NEON_GLOW_ALPHA, image, 1.0 - NEON_GLOW_ALPHA, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        anchor_x, anchor_y = x1, y1

    entity = display_id or (f"#{track_id}" if track_id > 0 else "")
    label = f"{entity} {category} {confidence * 100:.0f}%".strip()
    _draw_compact_label(image, anchor_x, anchor_y, label, color, text_color)


def draw_hud(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    latency_ms: float,
    show_legend: bool = True,
) -> np.ndarray:
    """Dibuja todos los frames arcoíris y la barra de estado AIoT sobre la imagen."""
    output = image.copy()

    for det in detections:
        draw_rainbow_frame(
            output,
            bbox=det["bbox"],
            category=det["category"],
            subtype=det.get("subtype", det["category"]),
            confidence=det["confidence"],
            track_id=det.get("track_id", 0),
            contour=det.get("contour"),
            display_id=det.get("display_id", ""),
        )

    h, w = output.shape[:2]

    # Barra superior semitransparente con métricas de Edge AIoT
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (18, 20, 28), -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)

    sla_ok = latency_ms <= TARGET_LATENCY_MS
    lat_color = (80, 235, 100) if sla_ok else (0, 165, 255)
    status_text = (
        f"Neuro_Trash_YL Edge AIoT | Objetos: {len(detections)} | "
        f"Latencia: {latency_ms:.1f} ms ({'SLA OK <180ms' if sla_ok else 'ALTA'})"
    )
    cv2.putText(
        output,
        status_text,
        (12, 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        lat_color,
        1,
        cv2.LINE_AA,
    )

    # Leyenda inferior de colores del arcoíris
    if show_legend and h > 160:
        legend_overlay = output.copy()
        bar_y1 = h - 34
        cv2.rectangle(legend_overlay, (0, bar_y1), (w, h), (18, 20, 28), -1)
        cv2.addWeighted(legend_overlay, 0.80, output, 0.20, 0, output)

        x_cursor = 10
        for cat_name, cat_meta in CATEGORIES.items():
            bgr = cat_meta["bgr"]
            cv2.rectangle(output, (x_cursor, bar_y1 + 9), (x_cursor + 14, bar_y1 + 23), bgr, -1)
            cv2.rectangle(output, (x_cursor, bar_y1 + 9), (x_cursor + 14, bar_y1 + 23), (240, 240, 240), 1)
            short_name = cat_name.replace("Desecho General", "General").replace("No Identificado", "No Ident.")
            cv2.putText(
                output,
                short_name,
                (x_cursor + 19, bar_y1 + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            (tw, _), _ = cv2.getTextSize(short_name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            x_cursor += tw + 34
            if x_cursor > w - 80:
                break

    return output
