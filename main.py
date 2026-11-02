"""
Neuro_Trash_YL - Script principal de consola y escritorio (CLI / OpenCV)
Alineado con la arquitectura Edge AIoT sobre Raspberry Pi 4/5 y el estilo de argumentos
(--camera, --stream-url, --vdo-url) para detección precisa con Frames Arcoíris.
"""

import argparse
from pathlib import Path
from typing import Union

import cv2

from config.settings import (
    CATEGORIES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_STABILITY_WINDOW,
    DEFAULT_UNKNOWN_THRESHOLD,
    PROJECT_DIR,
    TEST_IMAGES_DIR,
)
from src.camera_sources import is_vdo_ninja_url, open_video_source
from src.detector import NeuroTrashDetector
from src.telemetry import EdgeTelemetryManager
from src.visualizer import draw_hud


def detect_waste_in_images(
    image_dir: Path,
    detector: NeuroTrashDetector,
    telemetry: EdgeTelemetryManager,
) -> None:
    if not image_dir.exists():
        print(f"No existe la carpeta de imágenes: {image_dir}")
        return

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        print(f"Carga imágenes de prueba en la carpeta: {image_dir}")
        return

    print(f"Analizando {len(image_paths)} imagen(es) en {image_dir}. Pulsa cualquier tecla para avanzar o 'q' para salir.")
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"No se pudo leer: {image_path.name}")
            continue

        detector.reset_tracker()
        detections, latency_ms = detector.analyze_frame(image, enable_fallback_roi=True)
        payload = telemetry.record_frame_event(detections, latency_ms)
        annotated = draw_hud(image, detections, latency_ms, show_legend=True)

        summary = ", ".join(
            f"{d['category']} [{d['subtype']} {d['confidence']*100:.0f}% - {d['rainbow_color_name']}]"
            for d in detections
        ) or "Sin objetos detectados"
        print(
            f"[{image_path.name}] ({latency_ms:.1f} ms) -> {len(detections)} objeto(s): {summary} | "
            f"RPi GPIO PWM: {payload['raspberry_pi_gpio']['servo_angle_deg']}°"
        )

        cv2.imshow("Neuro_Trash_YL - Deteccion de Residuos (Frames Arcoiris)", annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def detect_waste_in_stream(
    source: Union[int, str],
    detector: NeuroTrashDetector,
    telemetry: EdgeTelemetryManager,
) -> None:
    capture = open_video_source(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente de video '{source}'. "
            "Si usas OBS Virtual Camera o una cámara USB/Raspberry Pi, verifica el índice (--camera 0, 1, 2). "
            "Si usas tu teléfono por IP Webcam/DroidCam, verifica la URL (ej. http://192.168.1.50:8080/video)."
        )

    print(f"Fuente de video activa ({source}). Pulsa 'q' en la ventana para salir.")
    while True:
        has_frame, frame = capture.read()
        if not has_frame or frame is None:
            print("No se pudo leer un fotograma de la fuente de video.")
            break

        detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=False)
        telemetry.record_frame_event(detections, latency_ms)
        annotated = draw_hud(frame, detections, latency_ms, show_legend=True)

        cv2.imshow("Neuro_Trash_YL - Reconocimiento de Residuos en Tiempo Real (Raspberry Pi Edge)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Neuro_Trash_YL: Detección y clasificación de materiales reciclables y basura orgánica "
            "con YOLO (Frames Arcoíris) desde cámara local/Raspberry Pi, vínculo móvil o imágenes."
        )
    )
    parser.add_argument(
        "--camera",
        type=int,
        metavar="INDICE",
        help="usa una cámara en tiempo real por índice; por ejemplo, --camera 0 (Webcam/CSI) o --camera 1 (OBS Virtual Camera)",
    )
    parser.add_argument(
        "--stream-url",
        metavar="URL",
        help="usa el vínculo directo de cámara del teléfono (ej. http://192.168.1.50:8080/video de IP Webcam o DroidCam)",
    )
    parser.add_argument(
        "--vdo-url",
        metavar="URL_VDO",
        help=(
            "documenta el enlace VDO.Ninja usado con el teléfono en OBS; "
            "recuerda abrirlo en OBS como Browser Source, activar OBS Virtual Camera y pasar --camera INDICE"
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=TEST_IMAGES_DIR,
        help=f"carpeta de imágenes de prueba cuando no se indica cámara (por defecto: {TEST_IMAGES_DIR})",
    )
    parser.add_argument(
        "--mode",
        choices=["world_hybrid", "fast_hybrid"],
        default="world_hybrid",
        help="modo del motor YOLO: 'world_hybrid' (máxima precisión semántica + material) o 'fast_hybrid' (ultra rápido para Raspberry Pi)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"umbral mínimo de detección para crear el frame del objeto (por defecto: {DEFAULT_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--unknown-thr",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
        help=f"umbral debajo del cual un objeto se marca como 'No Identificado' (por defecto: {DEFAULT_UNKNOWN_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("=" * 78)
    print(" NEURO_TRASH_YL | Sistema Inteligente de Detección Óptica y Telemetría AIoT")
    print(" Nodo de Borde: Raspberry Pi 4/5 | Paleta de Frames: 7 Colores del Arcoíris")
    print("=" * 78)
    for cat_name, meta in CATEGORIES.items():
        print(f"  - {cat_name:<16} -> {meta['rainbow_name']:<28} | {meta['container']}")
    print("=" * 78)

    if args.vdo_url:
        print(f"[VDO.Ninja] Fuente móvil configurada: {args.vdo_url}")
        print(
            "Nota: VDO.Ninja transmite por WebRTC. Ábrelo en OBS como 'Navegador' (Browser Source) "
            "o usa la Interfaz Web de Streamlit incluida en `app_web.py` para conectarlo directo desde el navegador."
        )

    detector = NeuroTrashDetector(
        detector_mode=args.mode,
        conf_threshold=args.conf,
        unknown_threshold=args.unknown_thr,
        use_stabilizer=(args.camera is not None or args.stream_url is not None),
        stability_window=DEFAULT_STABILITY_WINDOW,
    )
    telemetry = EdgeTelemetryManager()

    if args.stream_url:
        if is_vdo_ninja_url(args.stream_url) and args.camera is None:
            print(
                "Advertencia: El enlace proporcionado es de VDO.Ninja (WebRTC). "
                "Para usarlo con OpenCV activa OBS Virtual Camera y añade --camera <INDICE>, "
                "o ejecuta `streamlit run app_web.py` para integrarlo en la interfaz web."
            )
            return
        detect_waste_in_stream(args.stream_url, detector, telemetry)
        return

    if args.camera is not None:
        detect_waste_in_stream(args.camera, detector, telemetry)
        return

    detect_waste_in_images(args.images, detector, telemetry)


if __name__ == "__main__":
    main()
