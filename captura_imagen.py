"""
Captura fotos de la cámara USB (Razer Kiyo u otra) para fine-tune del clasificador.

Uso:
  .venv\\Scripts\\python.exe captura_imagen.py --clase plastic
  .venv\\Scripts\\python.exe captura_imagen.py --clase organic

Teclas: s = guardar, q = salir.
Meta recomendada: 80-150 fotos por clase, con distintos ángulos e iluminación.
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_sources import open_video_source

VALID_CLASSES = ("plastic", "metal", "paper", "cardboard", "glass", "trash", "organic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura fotos USB para reentrenar el clasificador.")
    parser.add_argument("--clase", default="plastic", choices=VALID_CLASSES, help="Carpeta de destino.")
    parser.add_argument("--kind", default="usb", choices=("usb", "obs"), help="Fuente: USB real u OBS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destino = ROOT / "dataset" / "capturas" / args.clase
    destino.mkdir(parents=True, exist_ok=True)

    source = "usb_auto" if args.kind == "usb" else "obs_auto"
    cap = open_video_source(source, allow_fallback_index=False, source_kind=args.kind)
    if not cap.isOpened():
        raise SystemExit(
            f"No se pudo abrir la fuente {args.kind}. "
            "Cierra Chrome/Streamlit si tienen la Kiyo ocupada y vuelve a intentar."
        )

    n = len(list(destino.glob("*.jpg")))
    print(f"Capturando {args.clase} en {destino} (ya hay {n} fotos). s=guardar, q=salir.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("La cámara dejó de enviar fotogramas.")
            break
        cv2.putText(
            frame,
            f"{args.clase}: {n}  (s=guardar, q=salir)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("captura", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            out = destino / f"{args.clase}_{n:04d}.jpg"
            cv2.imwrite(str(out), frame)
            n += 1
            print(f"Guardada {out.name}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
