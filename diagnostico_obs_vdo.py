"""
Herramienta de Diagnóstico y Reparación para OBS Virtual Camera + VDO.Ninja
Escanea todos los índices (0..6) y backends de Windows (DirectShow y MediaFoundation),
toma una captura de cada cámara encontrada, verifica si la imagen está en negro
y genera el enlace corregido de VDO.Ninja para OBS Browser Source.
"""

import argparse
import cv2
from src.camera_sources import (
    get_windows_video_device_names,
    normalize_vdo_ninja_url_for_obs,
    open_video_source,
    probe_available_cameras,
)


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de OBS Virtual Camera y VDO.Ninja")
    parser.add_argument("--vdo-url", default="", help="Pega tu enlace de VDO.Ninja para verificarlo y optimizarlo para OBS")
    parser.add_argument("--preview", type=int, help="Abre una ventana de prueba con el índice de cámara indicado")
    args = parser.parse_args()

    print("=" * 80)
    print(" DIAGNÓSTICO DE CÁMARA VIRTUAL OBS + VDO.NINJA (Neuro_Trash_YL)")
    print("=" * 80)

    if args.vdo_url:
        optimized = normalize_vdo_ninja_url_for_obs(args.vdo_url)
        print("\n[1] VERIFICACIÓN DE ENLACE VDO.NINJA:")
        print(f"  • Enlace original : {args.vdo_url}")
        print(f"  • Enlace para OBS : {optimized}")
        print("  👉 Copia el 'Enlace para OBS' y pégalo en las Propiedades de la fuente 'Navegador' en OBS.")

    print("\n[2] DISPOSITIVOS DE VIDEO DETECTADOS EN WINDOWS:")
    win_names = get_windows_video_device_names()
    if win_names:
        for i, name in enumerate(win_names):
            print(f"  • {name}")
    else:
        print("  • (No se pudieron listar nombres WMI, escaneando índices directamente...)")

    print("\n[3] ESCANEANDO ÍNDICES DE CÁMARA EN OPENCV (DSHOW + MSMF, Índices 0 al 5)...")
    cameras = probe_available_cameras(max_index=6)
    if not cameras:
        print("  ❌ No se encontró ninguna cámara abierta. Cierra otras apps que estén usando la cámara.")
    else:
        for cam in cameras:
            print(
                f"  --> Índice --camera {cam['index']} | Backend: {cam['backend']:<10} | "
                f"Resolución: {cam['resolution']:<9} | Brillo medio: {cam['brightness']:<5} | "
                f"{'⚠️ PANTALLA NEGRA' if cam['is_black'] else '✅ SEÑAL ACTIVA'}"
            )

    print("\n[4] SOLUCIÓN RÁPIDA SI VDO.NINJA NO SE VE EN OBS O EN PYTHON:")
    print("  1. En OBS: Ve a Archivo -> Ajustes -> Avanzado -> Fuentes -> DESMARCA 'Habilitar aceleración por hardware en el navegador' y reinicia OBS.")
    print("  2. En la fuente 'Navegador' de OBS: Asegúrate de que la URL diga '?view=...' (NO '?push=...') y añade '&cleanoutput&autostart'.")
    print("  3. En las propiedades de la fuente 'Navegador' de OBS: Haz clic en el botón 'Interactuar' (Interact) y toca 'Play' si el navegador bloqueó el autoplay.")
    print("  4. En OBS: Haz clic en 'Iniciar cámara virtual' (Start Virtual Camera) ANTES de abrir el script de Python o Streamlit.")
    print("=" * 80)

    if args.preview is not None:
        print(f"\nAbriendo vista previa del índice {args.preview}... (Pulsa 'q' para cerrar)")
        cap = open_video_source(args.preview)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("No se recibieron frames.")
                break
            cv2.imshow(f"Vista Previa Camara #{args.preview}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
