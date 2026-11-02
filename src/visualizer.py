"""
Renderizador visual de Frames con la paleta de colores del Arcoíris:
Dibuja cajas delimitadoras de alta visibilidad con esquinas reforzadas,
etiquetas legibles y panel HUD con telemetría Edge/IoT.
"""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from config.settings import CATEGORIES, TARGET_LATENCY_MS


def get_text_color_for_bg(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Calcula la luminancia del color de fondo para elegir texto blanco o negro."""
    b, g, r = bgr
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return (15, 15, 15) if luminance > 0.58 else (255, 255, 255)


def draw_rainbow_frame(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    category: str,
    subtype: str,
    confidence: float,
    track_id: int = 0,
) -> None:
    """Dibuja el frame (bounding box) con el color del arcoíris correspondiente a la categoría."""
    cat_info = CATEGORIES.get(category, CATEGORIES["No Identificado"])
    color = cat_info["bgr"]
    text_color = get_text_color_for_bg(color)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(2, x1), max(2, y1)
    x2, y2 = min(w - 2, x2), min(h - 2, y2)

    # Sombra exterior sutil para garantizar contraste sobre fondos blancos o muy oscuros
    cv2.rectangle(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (20, 20, 20), 4, cv2.LINE_AA)
    # Rectángulo principal con el color del arcoíris
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Esquinas reforzadas tipo visor inteligente
    box_w = x2 - x1
    box_h = y2 - y1
    corner_len = max(10, min(26, int(min(box_w, box_h) * 0.22)))
    thickness = 4
    # Superior-izquierda
    cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Superior-derecha
    cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
    # Inferior-izquierda
    cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
    # Inferior-derecha
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

    # Etiqueta superior con el tipo de material, sub-etiqueta y porcentaje de precisión
    id_prefix = f"#{track_id} " if track_id > 0 else ""
    label = f"{id_prefix}{category} | {subtype} ({confidence * 100:.0f}%)"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.52
    font_thick = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y1 = max(0, y1 - text_h - 10)
    label_y2 = y1 if y1 - text_h - 10 >= 0 else min(h, y1 + text_h + 10)
    label_x2 = min(w - 2, x1 + text_w + 12)

    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), (20, 20, 20), 1)
    text_y = label_y2 - 5
    cv2.putText(
        image,
        label,
        (x1 + 6, text_y),
        font,
        font_scale,
        text_color,
        font_thick,
        cv2.LINE_AA,
    )


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
