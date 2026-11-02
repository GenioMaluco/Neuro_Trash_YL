"""
Gestor universal de fuentes de video para Neuro_Trash_YL:
1. Soporta cámaras locales (USB / CSI en Raspberry Pi / OBS Virtual Camera).
2. Incluye respaldo automático de Captura de Ventana de OBS / VDO.Ninja en Windows
   (Win32 PrintWindow / GDI) cuando el driver DirectShow de OBS Virtual Camera no aparece
   como índice en OpenCV o cuando solo hay 1 cámara en el sistema (índice 0).
3. Soporta vínculos de cámara del teléfono (IP Webcam / DroidCam / MJPEG / RTSP).
"""

import ctypes
from ctypes import wintypes
import platform
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def is_vdo_ninja_url(source: str) -> bool:
    s = str(source).strip().lower()
    return "vdo.ninja" in s or "obs.ninja" in s


def normalize_vdo_ninja_url_for_obs(url: str) -> str:
    clean = url.strip()
    if not clean:
        return clean
    if "?push=" in clean:
        clean = clean.replace("?push=", "?view=")
    elif "&push=" in clean:
        clean = clean.replace("&push=", "&view=")

    sep = "&" if "?" in clean else "?"
    extra_params = []
    if "cleanoutput" not in clean.lower():
        extra_params.append("cleanoutput")
    if "autostart" not in clean.lower():
        extra_params.append("autostart")
    if extra_params:
        clean = f"{clean}{sep}{'&'.join(extra_params)}"
    return clean


def normalize_phone_stream_url(url: str) -> str:
    clean = url.strip()
    if clean.startswith(("http://", "https://")) and clean.rstrip("/").endswith(":8080"):
        return clean.rstrip("/") + "/video"
    return clean


# ============================================================================
# CAPTURA DIRECTA DE VENTANA DE OBS / NAVEGADOR VDO.NINJA EN WINDOWS (SIN DRIVER)
# Soluciona el error "No se pudo abrir la cámara con índice 1" cuando OBS Virtual Camera
# no es reconocida por OpenCV o está en otro índice.
# ============================================================================
class Win32WindowCapture:
    """
    Imita un objeto `cv2.VideoCapture` capturando en tiempo real la ventana de OBS
    (Proyector de OBS, ventana principal de OBS o pestaña de VDO.Ninja) usando la API
    nativa de Windows (`user32` / `gdi32`) sin instalar librerías adicionales.
    """

    def __init__(self, hwnd: int, title: str = ""):
        self.hwnd = hwnd
        self.title = title
        self._opened = bool(hwnd)

    def isOpened(self) -> bool:
        if not self._opened or platform.system() != "Windows":
            return False
        return bool(ctypes.windll.user32.IsWindow(self.hwnd))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.isOpened():
            return False, None

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        rect = wintypes.RECT()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            return False, None

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 32 or height <= 32:
            # Si la ventana está minimizada, obtener GetWindowRect
            user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width <= 32 or height <= 32:
                return False, None

        hwnd_dc = user32.GetWindowDC(self.hwnd)
        if not hwnd_dc:
            return False, None

        mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        save_bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        gdi32.SelectObject(mfc_dc, save_bitmap)

        # PW_RENDERFULLCONTENT = 0x00000002 permite capturar ventanas aceleradas por hardware (como OBS / Chrome)
        result = user32.PrintWindow(self.hwnd, mfc_dc, 2)
        if result == 0:
            user32.PrintWindow(self.hwnd, mfc_dc, 0)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height  # Top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        buffer_len = width * height * 4
        buffer = ctypes.create_string_buffer(buffer_len)
        gdi32.GetDIBits(mfc_dc, save_bitmap, 0, height, buffer, ctypes.byref(bmi), 0)

        gdi32.DeleteObject(save_bitmap)
        gdi32.DeleteDC(mfc_dc)
        user32.ReleaseDC(self.hwnd, hwnd_dc)

        img = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return True, frame_bgr

    def release(self) -> None:
        self._opened = False


def list_obs_and_vdo_windows() -> List[Dict[str, Union[int, str]]]:
    """Busca ventanas abiertas de OBS Studio o de VDO.Ninja en Windows."""
    if platform.system() != "Windows":
        return []

    user32 = ctypes.windll.user32
    found: List[Dict[str, Union[int, str]]] = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def enum_proc(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.strip()
                t_low = title.lower()
                if any(
                    kw in t_low
                    for kw in ("obs ", "obs studio", "proyector", "projector", "vdo.ninja", "obs.ninja")
                ):
                    found.append({"hwnd": int(hwnd), "title": title})
        return True

    user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
    # Priorizar ventanas de "Proyector" (Projector) de OBS si existen, luego OBS principal
    found.sort(key=lambda w: (0 if "proyector" in str(w["title"]).lower() or "projector" in str(w["title"]).lower() else 1))
    return found


def _try_open_camera_index(cam_idx: int) -> Optional[Tuple[cv2.VideoCapture, str]]:
    """
    Prueba abrir un índice de cámara en Windows probando tanto el backend por defecto
    (MSMF) como DirectShow (DSHOW), con y sin forzar MJPG.
    """
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, "V4L2/Default"
            cap.release()
        return None

    # Orden de intentos:
    # 1. DSHOW simple (como en el script original del usuario)
    # 2. Backend por defecto de OpenCV (MSMF - muchas veces OBS Virtual Camera solo abre aquí)
    # 3. DSHOW + MJPG 1280x720
    # 4. MSMF + MJPG 1280x720
    attempts = [
        (cv2.CAP_DSHOW, "DSHOW", None),
        (cv2.CAP_ANY, "AUTO/MSMF", None),
        (cv2.CAP_DSHOW, "DSHOW+MJPG", "MJPG"),
        (cv2.CAP_MSMF, "MSMF+MJPG", "MJPG"),
    ]

    for backend, b_name, fourcc_str in attempts:
        cap = cv2.VideoCapture(cam_idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if fourcc_str:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, b_name

        cap.release()

    return None


def open_video_source(source: Union[int, str], allow_fallback_index: bool = True):
    """
    Abre una fuente de video:
    - Si `source` es `"obs_window:<hwnd>"` o `"obs_auto"`, captura directamente la ventana de OBS / VDO.Ninja.
    - Si `source` es un índice entero (ej. 0, 1, 2):
      1) Intenta abrir ese índice con todos los backends disponibles (`DSHOW`, `MSMF`, `AUTO`).
      2) Si el usuario pidió el índice 1 (pensando que OBS Virtual Camera es la #1), pero en su PC
         no hay cámara física en el 0 (por lo que OBS Virtual Camera quedó en el índice 0),
         busca automáticamente en los índices 0, 2, 3 o en la ventana de OBS abierta para que NO falle.
    """
    if isinstance(source, str) and source.startswith("obs_window:"):
        hwnd = int(source.split(":", 1)[1])
        return Win32WindowCapture(hwnd=hwnd)

    if isinstance(source, str) and source == "obs_auto":
        obs_wins = list_obs_and_vdo_windows()
        if obs_wins:
            return Win32WindowCapture(hwnd=int(obs_wins[0]["hwnd"]), title=str(obs_wins[0]["title"]))

    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        cam_idx = int(source)
        opened = _try_open_camera_index(cam_idx)
        if opened is not None:
            return opened[0]

        # Si falló el índice solicitado (ej. pidió índice 1 pero OBS Virtual Camera está en el 0 o 2):
        if allow_fallback_index:
            for alt_idx in (0, 2, 3, 1):
                if alt_idx == cam_idx:
                    continue
                alt_opened = _try_open_camera_index(alt_idx)
                if alt_opened is not None:
                    return alt_opened[0]

            # Y si ningún índice de cámara abrió pero OBS o VDO.Ninja están abiertos en Windows,
            # capturar directamente la ventana de OBS / VDO.Ninja para que funcione sin bloquearse
            obs_wins = list_obs_and_vdo_windows()
            if obs_wins:
                return Win32WindowCapture(hwnd=int(obs_wins[0]["hwnd"]), title=str(obs_wins[0]["title"]))

        return cv2.VideoCapture(cam_idx)

    stream_url = str(source).strip()
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened() and stream_url.rstrip("/").endswith(":8080"):
        cap = cv2.VideoCapture(normalize_phone_stream_url(stream_url))
    return cap


def probe_available_cameras(max_index: int = 5) -> List[Dict[str, Union[int, str, bool]]]:
    """
    Escanea todas las cámaras disponibles (índices 0..max_index-1) Y además detecta
    cualquier ventana abierta de OBS Studio o VDO.Ninja como fuente seleccionable.
    """
    available: List[Dict[str, Union[int, str, bool]]] = []
    for idx in range(max_index):
        opened = _try_open_camera_index(idx)
        if opened is not None:
            cap, b_name = opened
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                mean_brightness = float(frame.mean())
                is_black = mean_brightness < 4.0
                status_note = "⚠️ Imagen Negra" if is_black else "✅ Con Señal"
                available.append(
                    {
                        "index": idx,
                        "backend": b_name,
                        "resolution": f"{w}x{h}",
                        "is_black": is_black,
                        "brightness": round(mean_brightness, 1),
                        "label": f"Cámara #{idx} ({b_name} {w}x{h}) — {status_note}",
                    }
                )
            cap.release()

    # Añadir también captura directa de ventanas de OBS / VDO.Ninja si están abiertas
    for win in list_obs_and_vdo_windows():
        available.append(
            {
                "index": f"obs_window:{win['hwnd']}",
                "backend": "Win32-Direct",
                "resolution": "Ventana",
                "is_black": False,
                "brightness": 100.0,
                "label": f"🖥️ Ventana Directa: {win['title'][:45]}",
            }
        )

    return available
