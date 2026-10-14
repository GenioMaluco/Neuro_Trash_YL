import cv2
from ultralytics import YOLOWorld

OBJETOS = [
    "computer mouse",
    "document",
    "AA battery",
    "100 Costa Rican colones coin",
]
NOMBRES = ["Mouse", "Papel", "Batería AA", "Moneda ₡100"]
UMBRAL_PAPEL = 0.05
UMBRAL_OTROS = 0.20
INDICE_MONEDA = 3
VALOR_MONEDA = 100

detector = YOLOWorld("yolov8s-world.pt")
detector.set_classes(OBJETOS)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Razer Kiyo X

while True:
    ok, frame = cap.read()
    if not ok:
        break

    detecciones = detector.predict(frame, imgsz=640, conf=UMBRAL_PAPEL, verbose=False)[0]
    conteo_monedas = 0
    if detecciones.boxes is not None:
        for caja in detecciones.boxes:
            indice = int(caja.cls[0])
            confianza = float(caja.conf[0])
            umbral = UMBRAL_PAPEL if indice == 1 else UMBRAL_OTROS
            if confianza < umbral:
                continue
            if indice == INDICE_MONEDA:
                conteo_monedas += 1
            nombre = NOMBRES[indice]
            x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            texto = f"{nombre} ({confianza*100:.0f}%)"
            cv2.putText(frame, texto, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    total_colones = conteo_monedas * VALOR_MONEDA
    cv2.putText(frame, f"Monedas: {conteo_monedas} | Total: ₡{total_colones}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Detector de objetos", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()