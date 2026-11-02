# ♻️ Neuro_Trash_YL — Sistema Inteligente de Detección Óptica (YOLO) y Telemetría IoT en Raspberry Pi

**Curso:** `87-0051` / `16-0700` — Integración de Soluciones de Internet de las Cosas, Datos e Inteligencia Artificial (ULACIT)  
**Proyecto de Investigación:** *Sistema Inteligente de Detección Óptica y Telemetría IoT para la Segregación Autónoma de Desechos en Tiempo Real*

**Equipo de Investigación:**
- **Adrián Araya Aguilar**
- **Julián Castro Montero**
- **Alberto Chaves Villegas**
- **Antony Solís Jara**

---

## 📋 1. Resumen Ejecutivo para la Exposición del Equipo

Este repositorio implementa el núcleo de **Visión Computacional en el Borde (Edge AI)** y **Telemetría IoT** de nuestra estación inteligente de clasificación de residuos. 

Durante la evolución del proyecto (documentada inicialmente en los **Entregables 1 al 5**), identificamos que los clasificadores tradicionales sobre imagen completa sufrían ante fondos complejos, variaciones de iluminación y múltiples objetos en escena, además de las limitaciones de memoria de la `ESP32-CAM`. 

En esta versión reestructurada se migró la arquitectura hacia una **Raspberry Pi 4 / 5** e implementamos un **Motor Híbrido de Detección y Clasificación con YOLOv8** capaz de:
1. **Encerrar cada objeto en un Frame preciso (*Bounding Box*)** ignorando a las personas y el mobiliario del fondo.
2. **Clasificar el material real sobre el recorte aislado del objeto** utilizando nuestro modelo entrenado (`runs/residuos/weights/best.pt` con **92.74% de exactitud Top-1**), fusionado con detección semántica para residuos orgánicos y especiales.
3. **Identificar visualmente la categoría mediante los 7 Colores del Arcoíris**, incluyendo una clase de seguridad (`No Identificado`) basada en *Muestreo por Incertidumbre (Active Learning / Human-in-the-Loop)*.
4. **Operar desde una Interfaz Web en Streamlit (`app_web.py`) o por CLI (`main.py`)**, permitiendo conectar cámaras USB/CSI, cámaras remotas desde el teléfono vía **VDO.Ninja + OBS Virtual Camera** o streams HTTP/MJPEG (*IP Webcam* / *DroidCam*).

---

## 🔄 2. Evolución Arquitectónica (Alineación con los Entregables del Curso)

Todos los avances escritos posteriores y el informe final (Semanas 10 a 15) se rigen por esta actualización técnica:

| Dimensión del Sistema | Diseño Inicial (Entregables 1–5) | Arquitectura Actual Implementada (`Neuro_Trash_YL`) | Beneficio Medible / Justificación |
| :--- | :--- | :--- | :--- |
| **Nodo de Borde (Edge/Fog)** | `ESP32-CAM` + PC auxiliar | **Raspberry Pi 4 / 5** (Linux ARM64 + Cámara USB/CSI + GPIO nativo) | Procesa redes profundas YOLO localmente sin depender de red Wi-Fi para accionar la compuerta mecánica. |
| **Modelo de IA** | `MobileNetV3 + PCA/SVD + KNN` sobre foto completa en fondo liso (600 imágenes) | **Pipeline YOLOv8 de 2 Etapas + Estabilizador IoU**:<br>1. Localización espacial (`YOLOv8s-World` / `YOLOv8n`)<br>2. Clasificación de material por recorte (`YOLOv8n-cls` entrenado en **2,527 imágenes**)<br>3. Votación temporal en ventana de *N* fotogramas | Aísla el objeto del fondo ruidoso, soporta múltiples objetos simultáneos y alcanza **92.74% Top-1 Accuracy** (`97%` papel, `96%` metal, `93%` plástico, `91%` cartón, `89%` vidrio). |
| **Codificación Visual** | 5 clases de texto plano | **7 Categorías con Frames de los Colores del Arcoíris** | Identificación inmediata en tiempo real para reciclables, compostaje orgánico y objetos no identificados. |
| **Actuación y Sensórica** | Simulación teórica de compuerta | **Controlador GPIO Raspberry Pi (`src/telemetry.py`)**:<br>• `GPIO 18 (PWM0)`: Servomotor de compuerta<br>• `GPIO 23/24`: Sensor ultrasónico `HC-SR04` de llenado | Cumple el SLA de tiempo real (**latencia `< 180 ms`**) y opera en modo *Gemelo Digital* cuando se ejecuta en PC. |
| **Interfaz de Usuario y Captura** | Script único de cámara local | **Interfaz Web en `Streamlit` (`app_web.py`) + CLI (`main.py`) + Diagnóstico OBS/VDO.Ninja** | Permite demostrar el sistema desde navegador web conectando el teléfono como cámara inalámbrica. |

---

## 🌈 3. Taxonomía de Residuos y Paleta Arcoíris de Frames

Cada objeto detectado recibe un recuadro (*Frame*) con esquinas reforzadas del color del arcoíris asignado a su material y posiciona el servomotor de la Raspberry Pi en el ángulo del contenedor correspondiente:

| Orden | Color del Arcoíris | Categoría | Ejemplos de Objetos Reconocidos | Ángulo Servo (`GPIO 18`) | Contenedor Destino |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | 🔴 **Rojo** (`#ED1C24`) | **`Plástico`** | Botellas PET, envases plásticos, vasos y bolsas | `30°` | Contenedor Azul/Rojo — Plásticos Reciclables |
| **2** | 🟠 **Naranja** (`#FF8C00`) | **`Metal`** | Latas de aluminio, hojalata, conservas, monedas | `60°` | Contenedor Amarillo — Metales y Latas |
| **3** | 🟡 **Amarillo** (`#FFE100`) | **`Papel/Cartón`** | Cajas de cartón, hojas de papel, periódicos, empaques | `90°` | Contenedor Gris/Azul — Papel y Cartón Limpio |
| **4** | 🟢 **Verde** (`#32CD32`) | **`Orgánico`** | Frutas, cáscaras (banano, manzana, cítricos), restos de comida | `120°` | Contenedor Verde — Compostaje / Biodegradable |
| **5** | 🔵 **Azul Cian** (`#00BFFF`) | **`Vidrio`** | Botellas, frascos, vasos y envases de vidrio | `150°` | Contenedor Verde Claro — Vidrio Reciclable |
| **6** | 🟣 **Índigo** (`#6A5ACD`) | **`Desecho General`** | Empaques metalizados/sucios, baterías AA, RAEE | `175°` | Contenedor Negro — No Valorizables / Especiales |
| **7** | 🟣 **Violeta** (`#D400FF`) | **`No Identificado`** | Objetos cuya confianza combinada es menor al umbral (`< 42%`) | `0°` *(Standby)* | Compuerta en espera — Evita contaminación cruzada |

---

## 🗂️ 4. Arquitectura de Carpetas y Módulos Realizados

```text
Neuro_Trash_YL/
├── app_web.py                 # Interfaz Web en Streamlit (Cámara local, Vínculo Móvil/VDO.Ninja, Telemetría RPi)
├── main.py                    # Punto de entrada CLI (--camera, --stream-url, --vdo-url, --images)
├── camara.py                  # Lanzador rápido retrocompatible con versiones previas
├── diagnostico_obs_vdo.py     # Utilidad de diagnóstico para OBS Virtual Camera y enlaces VDO.Ninja
├── entrenar_yolo.py           # Entrenamiento del modelo YOLOv8 de clasificación de materiales
├── evalua.py                  # Evaluación de métricas Top-1 Accuracy sobre validación
├── dividir_dataset.py         # Particionamiento 80/20 (train/val) del dataset de residuos
├── captura_imagen.py          # Herramienta de captura rápida de muestras por clase
├── config/
│   ├── settings.py            # Configuración central: 7 categorías Arcoíris, prompts YOLO, pines GPIO RPi
│   └── data_residuos.yaml     # Especificación de clases para entrenamiento de detección con cajas
├── src/
│   ├── detector.py            # Motor YOLO Híbrido de 2 etapas (Localización de Caja + Clasificador de Recorte)
│   ├── stabilizer.py          # Tracker Multi-Objeto por IoU + Votación temporal para suavizar Frames
│   ├── visualizer.py          # Renderizador OpenCV de Frames Arcoíris y barra HUD de latencia Edge
│   ├── camera_sources.py      # Gestor de cámaras (DSHOW/MSMF, OBS Virtual Cam, Win32 Window Capture, URLs)
│   └── telemetry.py           # Controlador GPIO de Raspberry Pi (gpiozero) y generador de JSON MQTT
└── runs/residuos/weights/
    └── best.pt                # Pesos entrenados del clasificador de materiales (92.74% Top-1 Accuracy)
```

---

## 🚀 5. Guía de Instalación y Ejecución

### Paso 1: Preparar el entorno virtual
```powershell
cd Neuro_Trash_YL
.\.env\Scripts\activate
pip install -r requirements.txt
```

### Paso 2: Ejecutar la Interfaz Web en Streamlit (Recomendado para demostraciones)
```powershell
streamlit run app_web.py
```
Dentro de la aplicación web podrás alternar en tiempo real entre:
1. **📷 Cámara Instalada (USB / CSI / OBS):** Escanea los índices del sistema o captura directamente la ventana de OBS Studio.
2. **📱 Vínculo de Sitio Web / Teléfono:**
   - **VDO.Ninja + OBS:** Pega tu vínculo `https://vdo.ninja/?view=TU_ID`, copia el enlace auto-optimizado (`&cleanoutput&autostart`) en una fuente *Navegador* (*Browser Source*) de OBS Studio, inicia la Cámara Virtual de OBS y activa el procesamiento en vivo.
   - **Stream Directo IP:** Conecta apps como *IP Webcam* (`http://192.168.x.x:8080/video`) o *DroidCam*.
3. **🖼️ Laboratorio de Imágenes:** Evalúa imágenes del repositorio (`dataset/img_test`) o sube fotografías propias para inspeccionar el payload MQTT y el ángulo del servomotor.

### Paso 3: Ejecutar desde Consola / CLI (`main.py`)
```powershell
# Evaluar las imágenes de prueba en dataset/img_test:
python main.py

# Abrir cámara local o cámara de Raspberry Pi (índice 0):
python main.py --camera 0

# Usar teléfono con VDO.Ninja a través de OBS Virtual Camera:
python main.py --vdo-url "https://vdo.ninja/?view=9MVEGaM" --camera 1

# Usar teléfono mediante vínculo HTTP/MJPEG directo:
python main.py --stream-url http://192.168.1.50:8080/video
```

---

## 🗺️ 6. Siguientes Pasos para Completar el Proyecto (Hoja de Ruta Semanas 10–15)

Para cerrar con éxito los entregables restantes del curso (**Avance 3 en Semana 13**, **Caso de Investigación** y **Entrega Final / Defensa en Semana 15**), el equipo seguirá este plan de trabajo:

### Fase 1: Ampliación del Dataset con Cajas Delimitadoras y Clase Orgánica Nativa (Semana 10–11)
- [ ] **Captura de muestras propias en Costa Rica:** Usar `captura_imagen.py` y herramientas de etiquetado (Roboflow / LabelImg) con el archivo `config/data_residuos.yaml` para agregar al menos **200 imágenes anotadas con *bounding boxes*** de residuos locales (incluyendo residuos **orgánicos** reales y empaques locales).
- [ ] **Ciclo de Aprendizaje Activo (*Active Learning*):** Recolectar los recortes que el sistema marque en color **Violeta (`No Identificado`)** durante las pruebas en cámara, etiquetarlos manualmente por el equipo y re-entrenar el modelo para demostrar empíricamente la mejora de precisión.

### Fase 2: Despliegue Físico en Raspberry Pi 4 / 5 y Maqueta Mecánica (Semana 11–12)
- [ ] **Montaje en Raspberry Pi:** Clonar la rama `develop` en la Raspberry Pi, instalar dependencias ARM64 y exportar los pesos de YOLO a formato **ONNX / NCNN** para reducir la latencia de inferencia por debajo de `100 ms`.
- [ ] **Cableado de pines GPIO (según `config/settings.py`):**
  - Conectar la señal PWM del servomotor de la compuerta al **`GPIO 18` (Pin físico 12)** con fuente externa de `5V`.
  - Conectar el sensor ultrasónico `HC-SR04` de nivel de llenado a **`GPIO 23` (`TRIG`)** y **`GPIO 24` (`ECHO`)** (con divisor resistivo a `3.3V`).
  - Conectar LEDs indicadores en **`GPIO 27`** (Verde: valorizable/orgánico) y **`GPIO 22`** (Rojo: alerta/no identificado).

### Fase 3: Integración del Broker MQTT con Node-RED / ThingsBoard (Semana 12–13 — Avance 3)
- [ ] **Conexión en vivo al Broker MQTT:** Configurar en `src/telemetry.py` la dirección del broker (`Mosquitto` local en la Raspberry Pi o broker en la nube) publicando en el tópico `ulacit/neuro_trash/estacion_01/telemetria`.
- [ ] **Dashboard en Node-RED / ThingsBoard:** Crear el panel visual que consuma el JSON emitido por `EdgeTelemetryManager` para graficar:
  - Conteo histórico por cada una de las 7 categorías del arcoíris.
  - Porcentaje de llenado en tiempo real de cada contenedor.
  - Alertas automáticas para el vehículo recolector cuando un contenedor supere el `80%` de capacidad (alineado con el **Entregable 5: Caso Aplicado**).

### Fase 4: Validación Experimental, Costos y Defensa Final (Semana 14–15)
- [ ] **Pruebas de protocolo físico:** Ejecutar 50 pruebas reales de segregación frente a la cámara midiendo:
  1. Exactitud por categoría (Matriz de confusión final).
  2. Latencia desglosada por capa: `Captura (ms)` + `Inferencia YOLO Edge (ms)` + `Actuación Servo GPIO (ms)` + `Publicación MQTT (ms)`.
- [ ] **Presupuesto y Retorno de Inversión (ROI):** Documentar en el informe final la tabla de costos del prototipo basado en Raspberry Pi + cámara estándar frente a sistemas ópticos industriales, destacando la viabilidad para campus universitarios (ULACIT), municipalidades e industria.
