"""Lienzo HITL: cada clic sobre el frame agrega un punto consecutivo del contorno."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from src.hitl import densify_points

Point = List[int]

_CLICK_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const img = parentElement.querySelector("#hitl-frame");
  if (!img) return;

  const onClick = (event) => {
    const rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height || !img.naturalWidth || !img.naturalHeight) {
      return;
    }
    const x = Math.round((event.clientX - rect.left) * (img.naturalWidth / rect.width));
    const y = Math.round((event.clientY - rect.top) * (img.naturalHeight / rect.height));
    setTriggerValue("click", { x: x, y: y });
  };

  img.onclick = onClick;
  img.alt = (data && data.alt) || "";
  if (data && data.src && img.getAttribute("src") !== data.src) {
    img.setAttribute("src", data.src);
  }

  return () => {
    img.onclick = null;
  };
}
"""

_CLICKABLE_FRAME = st.components.v2.component(
    "hitl_clickable_frame",
    html="""
<img id="hitl-frame" alt="" draggable="false" />
""",
    css="""
#hitl-frame {
  width: 100%;
  height: auto;
  display: block;
  cursor: crosshair;
  border-radius: 0.5rem;
  user-select: none;
}
""",
    js=_CLICK_JS,
)


def _bgr_to_data_url(frame_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _read_click(canvas_key: str) -> Optional[Dict[str, Any]]:
    state = st.session_state.get(canvas_key)
    if state is None:
        return None
    click = getattr(state, "click", None)
    if click is None and isinstance(state, dict):
        click = state.get("click")
    return click


def _append_click_point(display_id: str, canvas_key: str, width: int, height: int) -> None:
    click = _read_click(canvas_key)
    if not click:
        return
    try:
        x = int(click["x"] if isinstance(click, dict) else click.x)
        y = int(click["y"] if isinstance(click, dict) else click.y)
    except (AttributeError, KeyError, TypeError, ValueError):
        return
    x = max(0, min(max(width - 1, 0), x))
    y = max(0, min(max(height - 1, 0), y))
    points: List[Point] = [list(p) for p in (st.session_state.hitl_points.get(display_id) or [])]
    if points and points[-1] == [x, y]:
        return
    points.append([x, y])
    st.session_state.hitl_points[display_id] = points


def clear_hitl_points(display_id: str) -> None:
    st.session_state.hitl_points[display_id] = []


def restore_hitl_points(display_id: str) -> None:
    original = st.session_state.hitl_points_orig.get(display_id) or []
    st.session_state.hitl_points[display_id] = [list(p) for p in original]


def densify_hitl_points(display_id: str) -> None:
    current = st.session_state.hitl_points.get(display_id) or []
    if len(current) >= 2:
        st.session_state.hitl_points[display_id] = densify_points(current)


def clickable_mask_frame(
    frame_bgr: np.ndarray,
    *,
    display_id: str,
    key: str,
    alt: str,
) -> None:
    height, width = frame_bgr.shape[:2]
    _CLICKABLE_FRAME(
        key=key,
        data={"src": _bgr_to_data_url(frame_bgr), "alt": alt},
        width="stretch",
        on_click_change=lambda: _append_click_point(display_id, key, width, height),
    )
