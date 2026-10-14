import cv2
from pathlib import Path

CLASE = "metal"   # (carpeta de la clase)
DESTINO = Path("dataset/capturas") / CLASE
DESTINO.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Razer Kiyo X
n = len(list(DESTINO.glob("*.jpg")))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.putText(frame, f"{CLASE}: {n}  (s=guardar, q=salir)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("captura", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("s"):
        cv2.imwrite(str(DESTINO / f"{CLASE}_{n:04d}.jpg"), frame)
        n += 1
    elif k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()