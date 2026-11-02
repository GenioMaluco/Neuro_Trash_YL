"""
Neuro_Trash_YL - Interfaz Web Interactiva en Streamlit
Permite operar el sistema AIoT de reconocimiento de residuos con YOLO desde el navegador,
seleccionando:
  1. Cámara instalada en el equipo / Raspberry Pi (Webcam USB, CSI o OBS Virtual Camera)
  2. Vínculo / Sitio Web de Cámara desde el Teléfono (IP Webcam, DroidCam, VDO.Ninja + OBS o Cámara Web Móvil)
  3. Imágenes de prueba del repositorio o subidas por el usuario
Muestra los Frames con los 7 colores del Arcoíris, métricas de precisión, telemetría MQTT
y el estado de los pines GPIO de la Raspberry Pi 4/5.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from config.settings import (
    CATEGORIES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_STABILITY_WINDOW,
    DEFAULT_UNKNOWN_THRESHOLD,
    EDGE_ACTUATOR_GPIO_PINS,
    EDGE_NODE_HARDWARE,
    TARGET_LATENCY_MS,
    TEST_IMAGES_DIR,
)
from src.camera_sources import (
    is_vdo_ninja_url,
    normalize_phone_stream_url,
    normalize_vdo_ninja_url_for_obs,
    open_video_source,
    probe_available_cameras,
)
from src.detector import NeuroTrashDetector
from src.telemetry import EdgeTelemetryManager
from src.visualizer import draw_hud


st.set_page_config(
    page_title="Neuro_Trash_YL | Detección YOLO & Telemetría Raspberry Pi",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para las tarjetas de colores del arcoíris y panel de telemetría
st.markdown(
    """
    <style>
    .rainbow-badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.22);
    }
    .metric-card {
        background: rgba(28, 31, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelos YOLO (Detección + Clasificador de Materiales 92.7%)...")
def get_cached_detector(mode: str) -> NeuroTrashDetector:
    return NeuroTrashDetector(detector_mode=mode)


if "telemetry" not in st.session_state:
    st.session_state.telemetry = EdgeTelemetryManager()
if "detected_cameras" not in st.session_state:
    st.session_state.detected_cameras = [
        {"index": 0, "label": "Cámara #0 (Principal / OBS Virtual Camera si no hay otra webcam)"},
        {"index": 1, "label": "Cámara #1 (Secundaria / OBS Virtual Camera)"},
        {"index": 2, "label": "Cámara #2 (Auxiliar / Virtual)"},
        {"index": "obs_auto", "label": "🖥️ Captura Directa de Ventana OBS / VDO.Ninja (Sin Driver)"},
    ]


# ============================================================================
# BARRA LATERAL: CONFIGURACIÓN DE CÁMARA, VÍNCULO MÓVIL Y PRECISIÓN YOLO
# ============================================================================
with st.sidebar:
    st.title("♻️ Neuro_Trash_YL")
    st.caption("Nodo Edge: **Raspberry Pi 4 / 5** | Visión YOLOv8 Híbrida")

    st.subheader("1. Fuente de Video / Cámara")
    source_type = st.radio(
        "Selecciona el origen de captura:",
        options=[
            "📷 Cámara Instalada (USB / CSI / OBS)",
            "📱 Vínculo de Sitio Web / Teléfono (URL / VDO.Ninja)",
            "🖼️ Laboratorio de Imágenes / Pruebas",
        ],
        index=0,
    )

    st.divider()
    st.subheader("2. Ajustes de Precisión YOLO")
    detector_mode = st.selectbox(
        "Motor de Inferencia YOLO:",
        options=["world_hybrid", "fast_hybrid"],
        format_func=lambda m: (
            "🎯 YOLOv8-World + Clasificador Material (Máxima Precisión)"
            if m == "world_hybrid"
            else "⚡ YOLOv8n Edge + Clasificador Material (Ultra Rápido RPi)"
        ),
    )
    conf_threshold = st.slider(
        "Umbral de Detección de Caja (Frame):",
        min_value=0.10,
        max_value=0.85,
        value=float(DEFAULT_CONF_THRESHOLD),
        step=0.05,
        help="Nivel mínimo para que YOLO encierre un objeto en un Frame.",
    )
    unknown_threshold = st.slider(
        "Umbral de Certeza de Material:",
        min_value=0.20,
        max_value=0.85,
        value=float(DEFAULT_UNKNOWN_THRESHOLD),
        step=0.05,
        help="Si la confianza del material es menor a este valor, se marca con Frame Violeta ('No Identificado').",
    )
    stability_window = st.slider(
        "Estabilización Temporal (Frames):",
        min_value=1,
        max_value=10,
        value=int(DEFAULT_STABILITY_WINDOW),
        help="Votación entre N frames consecutivos para evitar que la caja parpadee.",
    )

    st.divider()
    st.subheader("🌈 Paleta Arcoíris de Frames")
    badges_html = ""
    for cat_name, meta in CATEGORIES.items():
        hex_c = meta["hex"]
        text_c = "#111111" if cat_name in {"Papel/Cartón", "Orgánico", "Vidrio"} else "#FFFFFF"
        badges_html += (
            f'<div class="rainbow-badge" style="background-color:{hex_c}; color:{text_c};">'
            f'{cat_name} • {meta["rainbow_name"].split(" ")[0]}'
            f"</div>"
        )
    st.markdown(badges_html, unsafe_allow_html=True)


# Inicializar y sincronizar parámetros del detector
detector = get_cached_detector(detector_mode)
detector.conf_threshold = conf_threshold
detector.unknown_threshold = unknown_threshold
detector.stabilizer.window_size = stability_window
telemetry: EdgeTelemetryManager = st.session_state.telemetry


# ============================================================================
# ENCABEZADO PRINCIPAL
# ============================================================================
st.markdown("## 🔬 Sistema Inteligente de Detección Óptica y Telemetría IoT para Segregación de Desechos")
st.markdown(
    "Reconocimiento en tiempo real con **Frames Arcoíris por tipo de material** "
    "(`Plástico`, `Metal`, `Papel/Cartón`, `Orgánico`, `Vidrio`, `Desecho General` y `No Identificado`) "
    "procesado en el borde (**Raspberry Pi**) con telemetría **MQTT**."
)

col_video, col_telemetry = st.columns([1.65, 1.0], gap="large")


def render_telemetry_panel(container, payload: dict, detections: list) -> None:
    """Actualiza el panel derecho con resultados de detección, pines GPIO de Raspberry Pi y JSON MQTT."""
    with container.container():
        st.subheader("📡 Estado del Nodo Edge (Raspberry Pi)")
        m1, m2, m3 = st.columns(3)
        lat = payload["edge_latency_ms"]
        sla_badge = "✅ En rango (<180ms)" if payload["meets_realtime_sla"] else "⚠️ Sobre 180ms"
        m1.metric("Latencia Edge", f"{lat:.1f} ms", sla_badge)
        m2.metric("Objetos Detectados", str(payload["objects_in_frame"]))
        m3.metric(
            "Servomotor RPi (GPIO 18)",
            f"{payload['raspberry_pi_gpio']['servo_angle_deg']}°",
            payload["raspberry_pi_gpio"]["status"],
        )

        st.markdown("#### 🎯 Objetos en el Frame Actual")
        if detections:
            for idx, det in enumerate(detections, start=1):
                hex_c = det["hex_color"]
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 7px solid {hex_c};">
                        <strong style="font-size:1.05rem;">#{idx} {det['category']} — {det['subtype']}</strong>
                        <span style="float:right; background:{hex_c}; color:#111; padding:2px 8px; border-radius:4px; font-weight:700;">
                            {det['confidence']*100:.1f}%
                        </span><br/>
                        <small>🎨 Frame: <b>{det['rainbow_color_name']}</b> | 🗑️ Destino: <b>{det['container']}</b></small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No se detectan residuos en este fotograma. Coloca un objeto frente a la cámara.")

        st.markdown("#### 📊 Nivel de Llenado de Contenedores (HC-SR04 GPIO 23/24)")
        for cat_name, fill_pct in payload["fill_levels_pct"].items():
            st.progress(int(min(100, fill_pct)), text=f"{cat_name}: {fill_pct:.1f}%")

        with st.expander("🔌 Ver Payload MQTT JSON (Node-RED / ThingsBoard)", expanded=False):
            st.code(telemetry.format_mqtt_json(payload), language="json")


# ============================================================================
# MODO 1: CÁMARA INSTALADA (USB / RASPBERRY PI CSI / OBS VIRTUAL CAMERA)
# ============================================================================
if source_type.startswith("📷"):
    with col_video:
        st.subheader("📷 Captura desde Cámara Instalada o Virtual (OBS)")
        c_scan, c_select = st.columns([1, 2])
        with c_scan:
            if st.button("🔍 Escanear cámaras del sistema", use_container_width=True):
                found = probe_available_cameras(max_index=5)
                if found:
                    st.session_state.detected_cameras = found
                    st.success(f"Se detectaron {len(found)} cámara(s).")
                else:
                    st.warning("No se detectaron cámaras activas en los índices 0..4.")

        with c_select:
            cam_options = st.session_state.detected_cameras
            selected_cam = st.selectbox(
                "Selecciona el índice o ventana de cámara:",
                options=cam_options,
                format_func=lambda item: item["label"],
            )
            cam_source = selected_cam["index"]

        run_live = st.toggle("▶️ Iniciar Detección en Tiempo Real (Frames Arcoíris)", value=False)
        frame_placeholder = st.empty()

    telemetry_placeholder = col_telemetry.empty()

    if run_live:
        detector.use_stabilizer = True
        cap = open_video_source(cam_source, allow_fallback_index=True)
        if not cap.isOpened():
            with col_video:
                st.error(
                    f"No se pudo abrir la fuente `{cam_source}`. "
                    "Haz clic en **🔍 Escanear cámaras del sistema** arriba para detectar automáticamente en qué índice está OBS o tu ventana de OBS."
                )
        else:
            while run_live:
                ok, frame = cap.read()
                if not ok or frame is None:
                    with col_video:
                        st.warning("No se pudo leer fotograma de la cámara.")
                    break

                detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=False)
                payload = telemetry.record_frame_event(detections, latency_ms)
                annotated = draw_hud(frame, detections, latency_ms, show_legend=True)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                frame_placeholder.image(
                    annotated_rgb,
                    caption=f"Fuente en vivo ({selected_cam['label']}) | Latencia Edge: {latency_ms:.1f} ms",
                    use_container_width=True,
                )
                render_telemetry_panel(telemetry_placeholder, payload, detections)
                time.sleep(0.01)
            cap.release()
    else:
        with col_video:
            st.info(
                "Activa el interruptor **▶️ Iniciar Detección en Tiempo Real** para abrir la cámara seleccionada, "
                "o usa la captura instantánea del navegador aquí abajo:"
            )
            snap = st.camera_input("📸 Captura rápida desde cámara del navegador")
            if snap is not None:
                file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                detector.use_stabilizer = False
                detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=True)
                payload = telemetry.record_frame_event(detections, latency_ms)
                annotated = draw_hud(frame, detections, latency_ms, show_legend=True)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                render_telemetry_panel(telemetry_placeholder, payload, detections)


# ============================================================================
# MODO 2: VÍNCULO DE SITIO WEB / TELÉFONO (IP WEBCAM, DROIDCAM, VDO.NINJA)
# ============================================================================
elif source_type.startswith("📱"):
    with col_video:
        st.subheader("📱 Conectar Cámara del Teléfono mediante Vínculo Web")
        phone_method = st.tabs(
            [
                "🔗 Stream Directo (IP Webcam / DroidCam / RTSP)",
                "🌐 Enlace VDO.Ninja (Como en tu proyecto facial)",
                "📲 Abrir Web desde el Teléfono",
            ]
        )

        with phone_method[0]:
            st.markdown(
                "Ingresa el vínculo HTTP/MJPEG o RTSP que te da la app de tu teléfono "
                "(por ejemplo, **IP Webcam** en Android: `http://192.168.1.50:8080/video` o **DroidCam**):"
            )
            stream_url = st.text_input(
                "Vínculo del stream del teléfono:",
                value="http://192.168.1.100:8080/video",
                placeholder="http://192.168.1.100:8080/video",
            )
            start_url_stream = st.toggle("▶️ Conectar y Analizar Stream del Teléfono", value=False)
            url_frame_placeholder = st.empty()

        with phone_method[1]:
            st.markdown(
                "Si estás usando **VDO.Ninja** (`https://vdo.ninja/?view=9MVEGaM`) y no logras verlo en OBS o en Python, "
                "usa este asistente automático o el **Visor Directo integrado**:"
            )
            vdo_url = st.text_input(
                "1️⃣ Vínculo VDO.Ninja (?view=...):",
                value="https://vdo.ninja/?view=9MVEGaM",
            )
            if vdo_url:
                fixed_vdo_url = normalize_vdo_ninja_url_for_obs(vdo_url)
                st.success("✅ **Enlace optimizado para OBS (cópialo y pégalo en tu fuente 'Navegador' de OBS):**")
                st.code(fixed_vdo_url, language="text")

                with st.expander("📺 Probar si tu teléfono está transmitiendo en vivo (Visor Directo VDO.Ninja)", expanded=False):
                    st.caption("Si ves el video de tu teléfono aquí abajo, tu enlace VDO.Ninja funciona bien y el bloqueo estaba en OBS:")
                    st.components.v1.iframe(fixed_vdo_url, height=340, scrolling=False)

            c_vdo1, c_vdo2 = st.columns([1, 1])
            with c_vdo1:
                if st.button("🔍 Escanear en qué índice está OBS Virtual Camera", use_container_width=True):
                    scanned = probe_available_cameras(max_index=6)
                    if scanned:
                        st.session_state.detected_cameras = scanned
                        for sc in scanned:
                            st.write(f"• {sc['label']}")
                    else:
                        st.warning("No se detectaron cámaras abiertas.")
            with c_vdo2:
                obs_cam_idx = st.number_input(
                    "2️⃣ Índice de OBS Virtual Camera en tu equipo:",
                    min_value=0,
                    max_value=6,
                    value=1,
                    step=1,
                )

            with st.expander("🛠️ ¿No se ve en OBS o en Python? Solución en 3 pasos", expanded=True):
                st.markdown(
                    """
                    1. **Pega el enlace optimizado en OBS:** En tu fuente **Navegador (Browser Source)** de OBS, usa el enlace de arriba que termina en **`&cleanoutput&autostart`** y define ancho `1280` y alto `720`.
                    2. **Si en OBS se ve la pantalla negra:**
                       - En las propiedades de la fuente **Navegador** en OBS, baja y marca **"Actualizar el navegador cuando la escena se active"**.
                       - Haz clic derecho sobre la fuente **Navegador** en OBS $\rightarrow$ **Interactuar (Interact)** y dale clic al centro por si el navegador espera confirmación de reproducción.
                       - Si sigue negro, ve en OBS a **Archivo $\rightarrow$ Ajustes $\rightarrow$ Avanzado $\rightarrow$ Fuentes** y **desmarca** *"Habilitar la aceleración por hardware en el navegador"*, luego reinicia OBS.
                    3. **Activa la Cámara Virtual en OBS:** Haz clic en **"Iniciar cámara virtual"** (*Start Virtual Camera*) en OBS, presiona el botón **🔍 Escanear** de arriba para ver cuál índice dice `1280x720 — ✅ Con Señal de Video`, pon ese número y activa el botón de abajo:
                    """
                )

            start_vdo_obs = st.toggle("▶️ Procesar VDO.Ninja vía OBS Virtual Camera", value=False)
            vdo_frame_placeholder = st.empty()

        with phone_method[2]:
            st.markdown(
                "También puedes abrir la dirección de esta app Streamlit directamente en el navegador de tu teléfono "
                "(conectado al mismo Wi-Fi de la PC o Raspberry Pi) y tomar/transmitir desde la cámara del celular:"
            )
            mobile_snap = st.camera_input("📸 Usar cámara del teléfono directamente en el navegador")

    telemetry_placeholder = col_telemetry.empty()

    if start_url_stream and stream_url:
        if is_vdo_ninja_url(stream_url):
            with col_video:
                st.warning(
                    "Detectamos un enlace de **VDO.Ninja** (WebRTC). "
                    "Usa la segunda pestaña **🌐 Enlace VDO.Ninja** para enlazarlo con OBS Virtual Camera."
                )
        else:
            normalized_url = normalize_phone_stream_url(stream_url)
            detector.use_stabilizer = True
            cap = open_video_source(normalized_url)
            if not cap.isOpened():
                with col_video:
                    st.error(
                        f"No se pudo conectar al vínculo `{normalized_url}`. "
                        "Verifica que tu teléfono y la PC/Raspberry Pi estén en la misma red Wi-Fi."
                    )
            else:
                while start_url_stream:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        with col_video:
                            st.warning("Fin del stream o pérdida de señal con el teléfono.")
                        break
                    detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=False)
                    payload = telemetry.record_frame_event(detections, latency_ms)
                    annotated = draw_hud(frame, detections, latency_ms, show_legend=True)
                    url_frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption=f"Stream Móvil ({normalized_url}) | Latencia: {latency_ms:.1f} ms",
                        use_container_width=True,
                    )
                    render_telemetry_panel(telemetry_placeholder, payload, detections)
                    time.sleep(0.01)
                cap.release()

    elif start_vdo_obs:
        detector.use_stabilizer = True
        cap = open_video_source(int(obs_cam_idx))
        if not cap.isOpened():
            with col_video:
                st.error(
                    f"No se pudo abrir OBS Virtual Camera en el índice {obs_cam_idx}. "
                    "Verifica que 'Iniciar Cámara Virtual' esté activo en OBS."
                )
        else:
            while start_vdo_obs:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=False)
                payload = telemetry.record_frame_event(detections, latency_ms)
                annotated = draw_hud(frame, detections, latency_ms, show_legend=True)
                vdo_frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"VDO.Ninja ({vdo_url or 'OBS Virtual Cam'}) | Latencia: {latency_ms:.1f} ms",
                    use_container_width=True,
                )
                render_telemetry_panel(telemetry_placeholder, payload, detections)
                time.sleep(0.01)
            cap.release()

    elif mobile_snap is not None:
        file_bytes = np.asarray(bytearray(mobile_snap.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        detector.use_stabilizer = False
        detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=True)
        payload = telemetry.record_frame_event(detections, latency_ms)
        annotated = draw_hud(frame, detections, latency_ms, show_legend=True)
        with col_video:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        render_telemetry_panel(telemetry_placeholder, payload, detections)


# ============================================================================
# MODO 3: LABORATORIO DE IMÁGENES / COMPARATIVA DE PRECISIÓN
# ============================================================================
else:
    telemetry_placeholder = col_telemetry.empty()
    with col_video:
        st.subheader("🖼️ Evaluación de Precisión sobre Imágenes de Prueba")
        test_files = sorted(
            p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ) if TEST_IMAGES_DIR.exists() else []

        img_source_opt = st.radio(
            "Origen de la imagen:",
            options=["Imágenes del repositorio (dataset/img_test)", "Subir una imagen propia"],
            horizontal=True,
        )

        selected_bgr = None
        caption_name = ""

        if img_source_opt.startswith("Imágenes") and test_files:
            chosen_path = st.selectbox(
                "Selecciona una imagen de prueba:",
                options=test_files,
                format_func=lambda p: p.name,
            )
            selected_bgr = cv2.imread(str(chosen_path))
            caption_name = chosen_path.name
        else:
            uploaded = st.file_uploader("Sube una fotografía de residuos (JPG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                selected_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                caption_name = uploaded.name

        if selected_bgr is not None:
            detector.use_stabilizer = False
            detector.reset_tracker()
            detections, latency_ms = detector.analyze_frame(selected_bgr, enable_fallback_roi=True)
            payload = telemetry.record_frame_event(detections, latency_ms)
            annotated = draw_hud(selected_bgr, detections, latency_ms, show_legend=True)

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Resultado con Frames Arcoíris: {caption_name} ({latency_ms:.1f} ms)",
                use_container_width=True,
            )
            render_telemetry_panel(telemetry_placeholder, payload, detections)
