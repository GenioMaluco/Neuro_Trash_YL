"""
Gestor universal de fuentes de video para Neuro_Trash_YL.
Las fuentes USB y OBS se catalogan por separado para no mezclarlas.

1. Soporta cámaras locales (USB / CSI en Raspberry Pi / OBS Virtual Camera).
2. Incluye respaldo automático de Captura de Ventana de OBS / VDO.Ninja en Windows
   (Win32 PrintWindow / GDI) cuando el driver DirectShow de OBS Virtual Camera no aparece
   como índice en OpenCV o cuando solo hay 1 cámara en el sistema (índice 0).
3. Soporta vínculos de cámara del teléfono (IP Webcam / DroidCam / MJPEG / RTSP).
"""

import ctypes
from ctypes import wintypes
import os
import platform
import subprocess
from typing import Dict, List, Optional, Tuple, Union

# Evita el fallo MSMF "can't grab frame" (-1072875772) en webcams USB de Windows.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2
import numpy as np


class ClosedCapture:
    """Sustituto de VideoCapture cuando no hay fuente usable."""

    def isOpened(self) -> bool:
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return False, None

    def release(self) -> None:
        return None


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


def is_virtual_device_name(name: str) -> bool:
    n = str(name).lower()
    return any(
        token in n
        for token in (
            "obs virtual",
            "obs-camera",
            "obs camera",
            "virtual camera",
            "remote desktop",
            "rdcamera",
            "manycam",
            "snap camera",
        )
    )


def looks_like_obs_placeholder(frame: Optional[np.ndarray]) -> bool:
    """Detecta la pantalla azul con logo de OBS Virtual Camera (sin señal)."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False
    h, w = frame.shape[:2]
    if h < 80 or w < 80:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 50, 15), (140, 255, 255))
    blue_ratio = float(np.count_nonzero(blue)) / blue.size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    logo = gray[int(h * 0.15) : int(h * 0.58), int(w * 0.32) : int(w * 0.68)]
    if logo.size == 0:
        return False
    return blue_ratio > 0.45 and float(logo.std()) > 35 and float(gray.mean()) < 90


def get_windows_video_device_names() -> List[str]:
    """Nombres de cámaras físicas y virtuales reportados por Windows."""
    if platform.system() != "Windows":
        return []
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.PNPClass -eq 'Camera' -or "
                "($_.PNPClass -eq 'MEDIA' -and $_.Name -match 'Camera|Webcam|Kiyo|OBS') } | "
                "Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return []
    names: List[str] = []
    seen = set()
    for line in completed.stdout.splitlines():
        name = line.strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def physical_usb_camera_names() -> List[str]:
    return [name for name in get_windows_video_device_names() if not is_virtual_device_name(name)]


def _read_usable_frame(cap, attempts: int = 8) -> Tuple[bool, Optional[np.ndarray]]:
    frame = None
    ok = False
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
    return False, None


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

    attempts = [
        (cv2.CAP_DSHOW, "DSHOW", None, None),
        (cv2.CAP_MSMF, "MSMF", None, None),
        (cv2.CAP_DSHOW, "DSHOW+MJPG", "MJPG", (1280, 720)),
        (cv2.CAP_MSMF, "MSMF+YUY2", "YUY2", (640, 480)),
        (cv2.CAP_MSMF, "MSMF+MJPG", "MJPG", (1280, 720)),
    ]

    for backend, b_name, fourcc_str, size in attempts:
        cap = cv2.VideoCapture(cam_idx, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if fourcc_str:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

        ok, _frame = _read_usable_frame(cap)
        if ok:
            return cap, b_name
        cap.release()

    return None


def _open_index_matching_kind(
    cam_idx: int,
    source_kind: str,
    obs_placeholder_only: bool = False,
) -> Optional[cv2.VideoCapture]:
    opened = _try_open_camera_index(cam_idx)
    if opened is None:
        return None
    cap, _backend = opened
    ok, frame = _read_usable_frame(cap)
    if not ok:
        cap.release()
        return None
    is_obs_feed = looks_like_obs_placeholder(frame)
    if source_kind == "usb" and is_obs_feed:
        cap.release()
        return None
    if source_kind == "obs" and obs_placeholder_only and not is_obs_feed:
        cap.release()
        return None
    return cap


def _open_first_kind_match(source_kind: str, skip_idx: Optional[int] = None, max_index: int = 6):
    for idx in range(max_index):
        if skip_idx is not None and idx == skip_idx:
            continue
        cap = _open_index_matching_kind(
            idx,
            source_kind,
            obs_placeholder_only=(source_kind == "obs"),
        )
        if cap is not None:
            return cap
    return None


def open_video_source(
    source: Union[int, str],
    allow_fallback_index: bool = False,
    source_kind: str = "auto",
):
    """
    Abre exactamente la fuente pedida.
    `source_kind='usb'` nunca cae a OBS. `source_kind='obs'` nunca abre una webcam USB.
    """
    if isinstance(source, str) and source.startswith("obs_window:"):
        hwnd = int(source.split(":", 1)[1])
        return Win32WindowCapture(hwnd=hwnd)

    if isinstance(source, str) and source == "obs_auto":
        obs_wins = list_obs_and_vdo_windows()
        if obs_wins:
            return Win32WindowCapture(hwnd=int(obs_wins[0]["hwnd"]), title=str(obs_wins[0]["title"]))
        cap = _open_first_kind_match("obs")
        return cap if cap is not None else ClosedCapture()

    if isinstance(source, str) and source == "usb_auto":
        cap = _open_first_kind_match("usb")
        return cap if cap is not None else ClosedCapture()

    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        cam_idx = int(source)
        kind = source_kind if source_kind in {"usb", "obs"} else "auto"
        if kind in {"usb", "obs"}:
            cap = _open_index_matching_kind(cam_idx, kind)
            if cap is not None:
                return cap
            if allow_fallback_index:
                cap = _open_first_kind_match(kind, skip_idx=cam_idx)
                if cap is not None:
                    return cap
            return ClosedCapture()

        opened = _try_open_camera_index(cam_idx)
        if opened is not None:
            return opened[0]
        if allow_fallback_index:
            cap = _open_first_kind_match("usb", skip_idx=cam_idx)
            if cap is not None:
                return cap
        return ClosedCapture()

    stream_url = str(source).strip()
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened() and stream_url.rstrip("/").endswith(":8080"):
        cap = cv2.VideoCapture(normalize_phone_stream_url(stream_url))
    return cap


def _camera_entry(
    index: Union[int, str],
    backend: str,
    frame: Optional[np.ndarray],
    kind: str,
    device_name: str = "",
) -> Dict[str, Union[int, str, bool, float]]:
    if frame is not None and frame.size > 0:
        h, w = frame.shape[:2]
        mean_brightness = float(frame.mean())
        resolution = f"{w}x{h}"
    else:
        mean_brightness = 0.0
        resolution = "—"
    is_placeholder = looks_like_obs_placeholder(frame)
    is_black = mean_brightness < 4.0
    if kind == "obs":
        prefix = device_name or "OBS Virtual Camera"
        status = "sin señal de escena" if is_placeholder or is_black else "con señal"
        label = f"{prefix} · índice {index} · {status}"
    else:
        prefix = device_name or f"Cámara USB #{index}"
        status = "ocupada / sin señal" if is_black else "lista"
        label = f"{prefix} · índice {index} · {status}"
    return {
        "index": index,
        "kind": kind,
        "backend": backend,
        "resolution": resolution,
        "is_black": is_black,
        "is_placeholder": is_placeholder,
        "brightness": round(mean_brightness, 1),
        "device_name": device_name,
        "label": label,
    }


def probe_available_cameras(max_index: int = 6) -> List[Dict[str, Union[int, str, bool]]]:
    """Escanea índices OpenCV y los clasifica como USB u OBS."""
    available: List[Dict[str, Union[int, str, bool]]] = []
    for idx in range(max_index):
        opened = _try_open_camera_index(idx)
        if opened is None:
            continue
        cap, b_name = opened
        ok, frame = _read_usable_frame(cap)
        if ok and frame is not None:
            kind = "obs" if looks_like_obs_placeholder(frame) else "usb"
            available.append(_camera_entry(idx, b_name, frame, kind))
        cap.release()

    for win in list_obs_and_vdo_windows():
        available.append(
            {
                "index": f"obs_window:{win['hwnd']}",
                "kind": "obs",
                "backend": "Win32-Direct",
                "resolution": "Ventana",
                "is_black": False,
                "is_placeholder": False,
                "brightness": 100.0,
                "device_name": str(win["title"]),
                "label": f"Ventana OBS: {str(win['title'])[:45]}",
            }
        )
    return available


def list_camera_catalog(max_index: int = 6) -> Dict[str, List[Dict[str, Union[int, str, bool]]]]:
    """Catálogo separado: cámaras USB físicas vs fuentes OBS."""
    catalog: Dict[str, List[Dict[str, Union[int, str, bool]]]] = {"usb": [], "obs": []}
    for cam in probe_available_cameras(max_index=max_index):
        catalog[str(cam.get("kind", "usb"))].append(cam)

    usb_names = physical_usb_camera_names()
    unnamed_usb = [c for c in catalog["usb"] if not c.get("device_name")]
    for cam, name in zip(unnamed_usb, usb_names):
        cam["device_name"] = name
        status = "ocupada / sin señal" if cam.get("is_black") else "lista"
        cam["label"] = f"{name} (USB) · índice {cam['index']} · {status}"
    has_open_usb = any(not c.get("is_placeholder") for c in catalog["usb"])
    if usb_names and not has_open_usb:
        for name in usb_names:
            catalog["usb"].append(
                {
                    "index": "usb_auto",
                    "kind": "usb",
                    "backend": "auto",
                    "resolution": "—",
                    "is_black": False,
                    "is_placeholder": False,
                    "brightness": 0.0,
                    "device_name": name,
                    "label": f"{name} (USB conectada)",
                }
            )
    elif not catalog["usb"]:
        catalog["usb"].append(
            {
                "index": "usb_auto",
                "kind": "usb",
                "backend": "auto",
                "resolution": "—",
                "is_black": False,
                "is_placeholder": False,
                "brightness": 0.0,
                "device_name": "",
                "label": "Buscar cámara USB automática",
            }
        )

    if not any(str(c.get("index", "")).startswith("obs") for c in catalog["obs"]):
        catalog["obs"].append(
            {
                "index": "obs_auto",
                "kind": "obs",
                "backend": "auto",
                "resolution": "—",
                "is_black": False,
                "is_placeholder": True,
                "brightness": 0.0,
                "device_name": "OBS",
                "label": "OBS Virtual Camera / ventana de OBS",
            }
        )
    return catalog
