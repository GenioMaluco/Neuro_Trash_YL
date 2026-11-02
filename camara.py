"""
Compatibilidad hacia atrás: redirige `camara.py` al nuevo motor unificado `main.py`
con Frames de colores del Arcoíris, detección híbrida YOLO y telemetría Raspberry Pi.
"""

from main import main

if __name__ == "__main__":
    import sys

    # Si se ejecuta `python camara.py` sin argumentos, abre la cámara 0 por defecto
    if len(sys.argv) == 1:
        sys.argv.extend(["--camera", "0"])
    main()
