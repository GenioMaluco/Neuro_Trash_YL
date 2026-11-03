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
├── app_web.py                 # Interfaz Web en Streamlit (cámara, teléfono, laboratorio HITL, telemetría RPi)
├── main.py                    # Punto de entrada CLI (--camera, --stream-url, --vdo-url, --images)
├── camara.py                  # Lanzador rápido retrocompatible con versiones previas
├── diagnostico_obs_vdo.py     # Utilidad de diagnóstico para OBS Virtual Camera y enlaces VDO.Ninja
├── entrenar_yolo.py           # Fine-tune del clasificador YOLOv8n-cls (materiales)
├── evalua.py                  # Evaluación Top-1 / Top-5 sobre dataset_yolo/val
├── dividir_dataset.py         # Mezcla TrashNet + capturas locales y parte 80/20 (train/val)
├── captura_imagen.py          # Captura rápida por clase desde USB u OBS (teclas s / q)
├── config/
│   ├── settings.py            # 7 categorías Arcoíris, prompts YOLO, pines GPIO, umbrales
│   └── data_residuos.yaml     # Clases para un futuro detector/segmentador nativo (dataset_objetos)
├── src/
│   ├── detector.py            # Motor YOLO híbrido de 2 etapas (caja/silueta + clasificador de recorte)
│   ├── stabilizer.py          # Tracker multi-objeto por IoU + votación temporal
│   ├── visualizer.py          # Frames Arcoíris, IDs de categoría y HUD de latencia
│   ├── camera_sources.py      # Cámaras USB, OBS Virtual Cam, captura de ventana y URLs
│   ├── telemetry.py           # GPIO Raspberry Pi (gpiozero) y payload MQTT
│   ├── hitl.py                # Catálogo ER, recortes, máscaras y lanzamiento del fine-tune
│   └── hitl_click_canvas.py   # Lienzo clicable: cada clic agrega un punto del contorno
├── dataset/
│   ├── Garbage classification/  # TrashNet público (plastic, metal, paper, cardboard, glass, trash)
│   ├── capturas/                # Recortes locales HITL / Kiyo (gitignorado)
│   ├── hitl/catalog.json        # Relación Categoría → Tipo de objeto → Muestra (gitignorado)
│   └── img_test/                # Imágenes de laboratorio / demostración
├── dataset_yolo/              # Dataset listo para Ultralytics (se regenera con dividir_dataset.py)
└── runs/residuos/weights/
    └── best.pt                # Pesos actuales del clasificador de materiales
```

---

## 🚀 5. Guía de Instalación y Ejecución

### Paso 1: Preparar el entorno virtual
```powershell
cd Neuro_Trash_YL
.\.env\Scripts\activate
pip install -r requirements.txt
```

### Paso 2: Ejecutar la Interfaz Web en Streamlit (Recomendado)
```powershell
streamlit run app_web.py
```

La barra lateral permite elegir el motor (`world_hybrid` = máxima precisión, `fast_hybrid` = más rápido en Raspberry Pi) y tres umbrales:

| Control | Qué hace |
| :--- | :--- |
| **Umbral de Detección de Caja** | Confianza mínima para encerrar un objeto. Bájalo si YOLO no ve el residuo. |
| **Umbral de Certeza de Material** | Si el clasificador queda por debajo, el objeto se marca **No Identificado** (magenta). |
| **Estabilización Temporal** | Vota la categoría durante *N* fotogramas para que el recuadro no parpadee. |

### Paso 3: Usar las tres fuentes de la herramienta

#### 📷 Cámara instalada (USB / CSI / OBS)
1. Elige **Cámara USB** o **Cámara OBS**. No se mezclan: USB nunca abre la cámara virtual de OBS.
2. Pulsa **Escanear dispositivos** si la webcam está ocupada (cierra Chrome, la app Cámara u otra pestaña de Streamlit).
3. Activa **Iniciar detección en tiempo real**.
4. Cuando el recuadro sea útil para entrenar, pulsa **Enviar frame a entrenamiento HITL**. Eso congela el fotograma y abre el laboratorio.

#### 📱 Teléfono (URL / VDO.Ninja)
- **Stream directo:** IP Webcam o DroidCam, por ejemplo `http://192.168.1.50:8080/video`. El teléfono y la PC/Raspberry Pi deben estar en la misma Wi-Fi.
- **VDO.Ninja + OBS:** pega `https://vdo.ninja/?view=TU_ID`, copia el enlace optimizado (`&cleanoutput&autostart`) en una fuente *Navegador* de OBS, inicia la Cámara Virtual y procesa desde el índice detectado.
- **Cámara del navegador:** abre la app Streamlit en el teléfono (misma red) y usa la pestaña *Abrir Web desde el Teléfono*.

#### 🖼️ Laboratorio e entrenamiento HITL
Sirve para corregir al modelo y generar nuevas muestras. Puedes partir de:

1. El **frame enviado** desde la cámara en vivo.
2. Una **foto del dispositivo** (`st.camera_input`).
3. Un **archivo subido**.
4. Una imagen de `dataset/img_test`.

Por cada objeto detectado (o agregado a mano):

1. Haz **clic sobre el frame** para dibujar el contorno en orden (mínimo 3 puntos). La línea sigue la silueta, no solo la caja.
2. Usa **Más puntos**, **Borrar** o **Restablecer**, o mueve un punto con los sliders X/Y.
3. Marca si la categoría es **Correcto** o **Corregir**.
4. Elige la **categoría del arcoíris** y el **tipo de objeto** (botella PET, lata, tetra pak, cáscara, etc.). Puedes crear un tipo nuevo.
5. En Papel/Cartón, indica si la clase de entrenamiento es `paper` o `cardboard`.
6. Pulsa **Guardar etiquetas y máscaras para entrenamiento**.

Cada muestra guardada produce:

- Recorte del objeto en `dataset/capturas/<clase>/` (lo que usa el clasificador).
- Máscara PNG del contorno.
- Entrada en `dataset/hitl/catalog.json` (`CAT-*` → `OBJ-*` → `SMP-*`).
- Imagen + etiqueta YOLO-seg en `dataset_objetos/` para un detector nativo futuro.

Luego, en **Catálogo ER y reentrenamiento**:

- **Preparar dataset_yolo** ejecuta `dividir_dataset.py`.
- **Reentrenar clasificador** prepara el dataset y lanza `entrenar_yolo.py` en segundo plano. El log queda en `dataset/hitl/train.log`.

### Paso 4: Ejecutar desde Consola / CLI (`main.py`)
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

## 🧠 6. Cómo Funciona el Entrenamiento del Modelo

El sistema es de **dos etapas**. Solo se reentrena de forma rutinaria la segunda:

| Etapa | Modelo | Qué aprende | Se reentrena con HITL |
| :--- | :--- | :--- | :---: |
| **1. Localizar** | `yolov8s-world.pt` (precisión) o `yolov8n.pt` (rápido) | Dónde está el objeto y su silueta | No. Usa prompts en `config/settings.py` |
| **2. Clasificar material** | `runs/residuos/weights/best.pt` (YOLOv8n-cls) | Si el recorte es plástico, metal, papel, cartón, vidrio, basura u orgánico | Sí |

Las clases entrenables son: `plastic`, `metal`, `paper`, `cardboard`, `glass`, `trash`, `organic`.

### 6.1 Ciclo recomendado (Human-in-the-Loop)

```text
Cámara en vivo  →  Enviar frame a HITL  →  Corregir contorno y categoría
        ↓
dataset/capturas/<clase>  +  catalog.json
        ↓
python dividir_dataset.py     (o el botón "Preparar dataset_yolo")
        ↓
python entrenar_yolo.py       (o el botón "Reentrenar clasificador")
        ↓
runs/residuos/weights/best.pt  →  python evalua.py  →  reiniciar Streamlit
```

Tras un fine-tune exitoso **reinicia la app** (`Ctrl+C` y `streamlit run app_web.py`) para que `@st.cache_resource` recargue `best.pt`.

### 6.2 Captura masiva por consola (opcional)

Si quieres muchas fotos de un solo material, sin pasar por el lienzo HITL:

```powershell
python captura_imagen.py --clase plastic
python captura_imagen.py --clase organic
python captura_imagen.py --clase metal --kind obs
```

Teclas: `s` guarda el fotograma en `dataset/capturas/<clase>/`, `q` sale. Meta útil: **80–150 fotos por clase**, con distintos ángulos, distancias e iluminación. Cierra Streamlit/Chrome si la webcam queda ocupada.

### 6.3 Armar `dataset_yolo`

```powershell
python dividir_dataset.py
```

El script:

1. Borra y recrea `dataset_yolo/`.
2. Mezcla **TrashNet** (`dataset/Garbage classification`) con las **capturas locales**.
3. Parte **80 % train / 20 % val**.
4. Si hay pocas fotos de la cámara real, las **repite en train** hasta ~80 por clase para que el fine-tune no ignore el dominio de la Kiyo/Raspberry Pi.

No hace falta commitear `dataset_yolo` ni `dataset/capturas/`: se regeneran en cada máquina.

### 6.4 Fine-tune del clasificador

```powershell
python entrenar_yolo.py
```

Parte de `runs/residuos/weights/best.pt` si existe; si no, de `yolov8n-cls.pt`. Hiperparámetros actuales:

- **50 épocas**, `imgsz=256`, `batch=32`, `patience=15`
- Aumentos: HSV, volteo horizontal, rotación ±12°, traslación y escala
- Salida: `runs/residuos/` (sobrescribe el run anterior con `exist_ok=True`)

En la app, el botón **Reentrenar clasificador** hace lo mismo en segundo plano. Si ya hay un job, verás *en curso* hasta que termine.

### 6.5 Evaluar el modelo

```powershell
python evalua.py
```

Imprime Top-1 y Top-5 sobre `dataset_yolo/val`. Úsalo después de cada fine-tune para comprobar que el material local (cámara real) mejoró sin destruir el desempeño en TrashNet.

### 6.6 Qué no se reentrena todavía

El **localizador** (YOLO-World / YOLOv8n) no se fine-tunea en este ciclo. Si un objeto nunca aparece, primero baja el umbral de caja o amplía `WORLD_WASTE_PROMPTS` en `config/settings.py`. Las máscaras HITL ya se guardan en `dataset_objetos/` + `config/data_residuos.yaml` para, en una fase posterior, entrenar un detector/segmentador nativo de residuos.

---

## 🗺️ 7. Siguientes Pasos para Completar el Proyecto (Hoja de Ruta Semanas 10–15)

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
