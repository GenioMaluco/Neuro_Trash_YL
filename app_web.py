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

import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

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
    list_camera_catalog,
    normalize_phone_stream_url,
    normalize_vdo_ninja_url_for_obs,
    open_video_source,
    physical_usb_camera_names,
    probe_available_cameras,
)
from src.detector import NeuroTrashDetector
from src.hitl import (
    REVIEWABLE_CATEGORIES,
    catalog_stats,
    infer_train_class,
    load_catalog,
    prepare_dataset,
    retrain_running,
    save_reviews,
    start_retrain,
)
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
        border: 1px solid rgba(255,255,255,0.45);
        box-shadow: 0 0 10px currentColor;
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
def get_cached_detector(mode: str, _prompt_pack: str = "v2") -> NeuroTrashDetector:
    return NeuroTrashDetector(detector_mode=mode)


if "telemetry" not in st.session_state:
    st.session_state.telemetry = EdgeTelemetryManager()
st.session_state.setdefault("camera_catalog", {"usb": [], "obs": []})
st.session_state.setdefault("cameras_scanned", False)
st.session_state.setdefault("detected_cameras", [])


def _release_live_cap() -> None:
    cap = st.session_state.pop("live_cap", None)
    st.session_state.pop("live_cap_key", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _get_live_cap(source, source_kind: str):
    key = (str(source), source_kind)
    cap = st.session_state.get("live_cap")
    if cap is not None and st.session_state.get("live_cap_key") == key and cap.isOpened():
        return cap
    _release_live_cap()
    cap = open_video_source(source, allow_fallback_index=False, source_kind=source_kind)
    st.session_state.live_cap = cap
    st.session_state.live_cap_key = key
    return cap


def _on_run_live_change() -> None:
    if not st.session_state.get("run_live"):
        _release_live_cap()


def _refresh_camera_catalog() -> None:
    catalog = list_camera_catalog(max_index=6)
    st.session_state.camera_catalog = catalog
    st.session_state.detected_cameras = catalog["usb"] + catalog["obs"]
    st.session_state.cameras_scanned = True


@st.fragment(run_every=0.12)
def render_live_local_camera():
    source = st.session_state.live_source
    kind = st.session_state.live_kind
    label = st.session_state.live_label
    frame_slot = st.session_state.live_frame_slot
    telemetry_slot = st.session_state.live_telemetry_slot
    cap = _get_live_cap(source, kind)
    if not cap.isOpened():
        usb_names = physical_usb_camera_names()
        if kind == "usb":
            extra = f" Windows ve: {', '.join(usb_names)}." if usb_names else ""
            frame_slot.error(
                "No se pudo abrir la cámara USB seleccionada." + extra
                + " Cierra otras pestañas de esta app, Chrome o la app Cámara, y pulsa Escanear."
            )
        else:
            frame_slot.error(
                "No se pudo abrir la fuente OBS. Inicia la cámara virtual en OBS o deja visible la ventana de OBS."
            )
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        frame_slot.warning("La fuente seleccionada no está enviando fotogramas.")
        _release_live_cap()
        return

    live_detector = get_cached_detector(st.session_state.get("live_detector_mode", "world_hybrid"))
    live_detector.conf_threshold = st.session_state.get("live_conf", live_detector.conf_threshold)
    live_detector.unknown_threshold = st.session_state.get("live_unknown", live_detector.unknown_threshold)
    live_detector.stabilizer.window_size = st.session_state.get(
        "live_stability", live_detector.stabilizer.window_size
    )
    live_detector.use_stabilizer = True
    detections, latency_ms = live_detector.analyze_frame(frame, enable_fallback_roi=True)
    payload = st.session_state.telemetry.record_frame_event(detections, latency_ms)
    annotated = draw_hud(frame, detections, latency_ms, show_legend=True)
    frame_slot.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption=f"{label} · {latency_ms:.1f} ms",
        width="stretch",
        alt=f"Detección en vivo desde {label}",
    )
    render_telemetry_panel(telemetry_slot, payload, detections)


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
            "🖼️ Laboratorio e entrenamiento HITL",
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
        text_c = meta.get("label_fg", "#FFFFFF")
        badges_html += (
            f'<div class="rainbow-badge" style="background-color:{hex_c}; color:{text_c}; '
            f'text-shadow: 0 0 8px {hex_c};">'
            f'{meta["category_id"]} · {cat_name}'
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
                display_id = det.get("display_id", f"#{idx}")
                category_id = det.get("category_id", "")
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 7px solid {hex_c}; box-shadow: 0 0 12px {hex_c}55;">
                        <strong style="font-size:1.05rem;">{display_id} {det['category']} — {det['subtype']}</strong>
                        <span style="float:right; background:{hex_c}; color:#111; padding:2px 8px; border-radius:4px; font-weight:700;">
                            {det['confidence']*100:.1f}%
                        </span><br/>
                        <small>ID: <b>{category_id}</b> | Frame: <b>{det['rainbow_color_name']}</b> | Destino: <b>{det['container']}</b></small>
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
            st.code(st.session_state.telemetry.format_mqtt_json(payload), language="json")


# ============================================================================
# MODO 1: CÁMARA USB O CÁMARA OBS (FUENTES SEPARADAS)
# ============================================================================
if source_type.startswith("📷"):
    if not st.session_state.cameras_scanned:
        with st.spinner("Buscando cámaras USB y fuentes OBS..."):
            _refresh_camera_catalog()

    with col_video:
        st.subheader("Captura desde cámara USB u OBS")
        camera_kind = st.segmented_control(
            "Tipo de fuente",
            options=["usb", "obs"],
            format_func=lambda kind: "Cámara USB" if kind == "usb" else "Cámara OBS",
            default="usb",
            required=True,
            key="camera_kind",
            width="stretch",
        )
        c_scan, c_select = st.columns([1, 2])
        with c_scan:
            if st.button("Escanear dispositivos", width="stretch"):
                _release_live_cap()
                _refresh_camera_catalog()
                st.success("Catálogo de cámaras actualizado.")

        catalog = st.session_state.camera_catalog
        cam_options = catalog.get(camera_kind, [])
        with c_select:
            if cam_options:
                labels = [item["label"] for item in cam_options]
                selected_label = st.selectbox(
                    "Dispositivo de esta fuente",
                    options=labels,
                    key=f"selected_{camera_kind}_label",
                )
                selected_cam = next(item for item in cam_options if item["label"] == selected_label)
            else:
                selected_cam = {
                    "index": "usb_auto" if camera_kind == "usb" else "obs_auto",
                    "label": "Fuente automática",
                    "kind": camera_kind,
                }
                st.warning("No hay dispositivos en esta categoría. Pulsa Escanear dispositivos.")

        if camera_kind == "usb":
            usb_names = physical_usb_camera_names()
            if usb_names:
                st.caption("USB detectadas por Windows: " + ", ".join(usb_names))
            open_usb = [c for c in catalog.get("usb", []) if c.get("index") not in {"usb_auto"}]
            if usb_names and not open_usb:
                st.warning(
                    "La cámara USB está conectada, pero otra app la está usando. "
                    "Cierra Chrome u otras pestañas de esta app y vuelve a escanear."
                )
        else:
            st.caption("Esta opción usa OBS Virtual Camera o la ventana de OBS, nunca la webcam USB.")

        run_live = st.toggle(
            "Iniciar detección en tiempo real",
            key="run_live",
            on_change=_on_run_live_change,
        )
        frame_placeholder = st.empty()

    telemetry_placeholder = col_telemetry.empty()
    frame_placeholder.caption("Vista de la fuente seleccionada")
    telemetry_placeholder.caption("Telemetría")

    st.session_state.live_source = selected_cam["index"]
    st.session_state.live_kind = camera_kind
    st.session_state.live_label = selected_cam["label"]
    st.session_state.live_frame_slot = frame_placeholder
    st.session_state.live_telemetry_slot = telemetry_placeholder
    st.session_state.live_detector_mode = detector_mode
    st.session_state.live_conf = conf_threshold
    st.session_state.live_unknown = unknown_threshold
    st.session_state.live_stability = stability_window

    if run_live:
        render_live_local_camera()
    else:
        _release_live_cap()
        with col_video:
            st.info(
                "Elige **Cámara USB** o **Cámara OBS** y activa la detección. "
                "La vista mostrará solo la fuente seleccionada."
            )


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
                    detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=True)
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
        cap = open_video_source(int(obs_cam_idx), allow_fallback_index=False, source_kind="obs")
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
                detections, latency_ms = detector.analyze_frame(frame, enable_fallback_roi=True)
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
# MODO 3: LABORATORIO + HUMAN-IN-THE-LOOP
# ============================================================================
else:
    telemetry_placeholder = col_telemetry.empty()
    with col_video:
        st.subheader("Laboratorio e entrenamiento human-in-the-loop")
        st.caption(
            "Sube o elige una foto, revisa cada ID (CAT → OBJ → SMP) y guarda "
            "la etiqueta correcta para que el clasificador aprenda esos objetos."
        )
        test_files = sorted(
            p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ) if TEST_IMAGES_DIR.exists() else []

        img_source_opt = st.segmented_control(
            "Origen de la imagen",
            options=["upload", "repo"],
            format_func=lambda opt: (
                "Subir imagen" if opt == "upload" else "Imágenes del repositorio"
            ),
            default="upload",
            required=True,
            key="hitl_image_source",
        )

        selected_bgr = None
        caption_name = ""

        if img_source_opt == "repo" and test_files:
            chosen_path = st.selectbox(
                "Selecciona una imagen de prueba",
                options=test_files,
                format_func=lambda p: p.name,
            )
            selected_bgr = cv2.imread(str(chosen_path))
            caption_name = chosen_path.name
        elif img_source_opt == "repo":
            st.warning("No hay fotos en dataset/img_test.")
        else:
            uploaded = st.file_uploader(
                "Sube una fotografía de residuos",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                selected_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                caption_name = uploaded.name

        if selected_bgr is not None:
            fingerprint = hashlib.md5(
                f"{caption_name}:{selected_bgr.shape}:{selected_bgr.mean():.2f}".encode()
            ).hexdigest()[:10]
            if st.session_state.get("hitl_fingerprint") != fingerprint:
                detector.use_stabilizer = False
                detector.reset_tracker()
                detections, latency_ms = detector.analyze_frame(
                    selected_bgr, enable_fallback_roi=True
                )
                st.session_state.hitl_fingerprint = fingerprint
                st.session_state.hitl_detections = detections
                st.session_state.hitl_latency = latency_ms
                st.session_state.hitl_frame = selected_bgr
                st.session_state.hitl_name = caption_name

            detections = st.session_state.hitl_detections
            latency_ms = st.session_state.hitl_latency
            payload = telemetry.record_frame_event(detections, latency_ms)
            annotated = draw_hud(selected_bgr, detections, latency_ms, show_legend=True)

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"{caption_name} · {latency_ms:.1f} ms",
                width="stretch",
                alt=f"Detección anotada de {caption_name} con IDs de categoría",
            )

            if not detections:
                st.info("No se detectó ningún objeto. Prueba otra foto o baja el umbral de caja.")
            else:
                st.markdown("#### Revisión por objeto")
                st.caption("Cada fila es una instancia detectada. El ID (PL-01) apunta a su categoría (CAT-PL).")
                with st.form("hitl_review_form"):
                    reviews = []
                    for det in detections:
                        display_id = det["display_id"]
                        predicted = det["category"] if det["category"] in REVIEWABLE_CATEGORIES else "Plástico"
                        hex_c = det.get("hex_color", "#FF1744")
                        with st.container(border=True):
                            thumb_col, fields_col = st.columns([1, 2.4], vertical_alignment="top")
                            crop = selected_bgr[
                                max(0, det["bbox"][1]) : max(0, det["bbox"][3]),
                                max(0, det["bbox"][0]) : max(0, det["bbox"][2]),
                            ]
                            with thumb_col:
                                if crop.size > 0:
                                    st.image(
                                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                        width="stretch",
                                        alt=f"Recorte {display_id}",
                                    )
                                st.caption(f"{display_id} · {det.get('category_id')}")
                            with fields_col:
                                st.markdown(
                                    f"**{display_id}** predijo **{det['category']}** "
                                    f"({det['subtype']}, {det['confidence'] * 100:.0f}%)"
                                )
                                verdict = st.segmented_control(
                                    "¿La categoría es correcta?",
                                    options=["Correcto", "Corregir"],
                                    default="Correcto" if predicted == det["category"] and det["category"] != "No Identificado" else "Corregir",
                                    key=f"hitl_verdict_{fingerprint}_{display_id}",
                                    required=True,
                                )
                                category = st.selectbox(
                                    "Categoría",
                                    options=REVIEWABLE_CATEGORIES,
                                    index=REVIEWABLE_CATEGORIES.index(predicted),
                                    key=f"hitl_cat_{fingerprint}_{display_id}",
                                )
                                object_name = st.text_input(
                                    "Nombre del objeto",
                                    value=det.get("object_name") or det.get("subtype") or category,
                                    key=f"hitl_name_{fingerprint}_{display_id}",
                                )
                                train_default = infer_train_class(category, object_name)
                                paper_opts = ("paper", "cardboard")
                                if category == "Papel/Cartón":
                                    train_class = st.segmented_control(
                                        "Clase de entrenamiento",
                                        options=paper_opts,
                                        default=train_default if train_default in paper_opts else "paper",
                                        key=f"hitl_train_{fingerprint}_{display_id}",
                                        required=True,
                                    )
                                else:
                                    train_class = train_default
                        reviews.append(
                            {
                                "display_id": display_id,
                                "verdict": "correct" if verdict == "Correcto" else "corrected",
                                "category": category,
                                "object_name": object_name,
                                "train_class": train_class,
                                "hex": hex_c,
                            }
                        )
                    submitted = st.form_submit_button(
                        "Guardar etiquetas para entrenamiento",
                        icon=":material/save:",
                        type="primary",
                        width="stretch",
                    )

                if submitted:
                    stored = save_reviews(
                        selected_bgr,
                        detections,
                        reviews,
                        source_name=caption_name,
                    )
                    if stored:
                        st.success(
                            f"Guardadas {len(stored)} muestras. "
                            + ", ".join(f"{s['sample_id']}→{s['object_type_id']} ({s['train_class']})" for s in stored)
                        )
                    else:
                        st.warning("No se guardó ninguna muestra. Revisa que la categoría no sea 'No Identificado'.")

            render_telemetry_panel(telemetry_placeholder, payload, detections)
        else:
            telemetry_placeholder.info("Sube o selecciona una imagen para revisar objetos.")

    stats = catalog_stats()
    st.divider()
    st.subheader("Catálogo ER y reentrenamiento")
    s1, s2, s3 = st.columns(3)
    s1.metric("Tipos de objeto (OBJ)", str(stats["object_types"]))
    s2.metric("Muestras (SMP)", str(stats["samples"]))
    s3.metric("Reentrenamiento", "en curso" if retrain_running() else "en espera")

    catalog = load_catalog()
    if catalog["object_types"]:
        with st.expander("Ver relaciones categoría → objeto → muestra", expanded=False):
            type_by_id = {item["object_type_id"]: item for item in catalog["object_types"]}
            rows = []
            for sample in catalog["samples"][-40:]:
                obj = type_by_id.get(sample["object_type_id"], {})
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "object_type_id": sample["object_type_id"],
                        "object_name": obj.get("name", sample.get("object_name")),
                        "category_id": sample["category_id"],
                        "train_class": sample["train_class"],
                        "verdict": sample["verdict"],
                        "source": sample["source_name"],
                    }
                )
            st.dataframe(rows, width="stretch", alt="Muestras HITL con IDs de categoría, objeto y recorte")

    train_col, prep_col = st.columns(2)
    with prep_col:
        if st.button("Preparar dataset_yolo", icon=":material/folder_copy:", width="stretch"):
            result = prepare_dataset()
            if result.returncode == 0:
                st.success(result.stdout or "dataset_yolo listo.")
            else:
                st.error(result.stderr or result.stdout or "Falló dividir_dataset.py")
    with train_col:
        if st.button(
            "Reentrenar clasificador",
            icon=":material/model_training:",
            width="stretch",
            disabled=retrain_running(),
        ):
            prep = prepare_dataset()
            if prep.returncode != 0:
                st.error(prep.stderr or prep.stdout or "No se pudo preparar el dataset.")
            else:
                try:
                    log_path = start_retrain()
                    st.success(f"Fine-tune iniciado. Log: {log_path}")
                except RuntimeError as exc:
                    st.warning(str(exc))
