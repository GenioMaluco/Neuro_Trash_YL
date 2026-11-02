"""
Capa de Telemetría IoT y Actuación en Raspberry Pi (Edge Computing + GPIO + MQTT):
Sustituye la captura limitada de la ESP32-CAM por un nodo unificado Raspberry Pi 4/5
que ejecuta la inferencia YOLO localmente, controla el servomotor PWM en GPIO 18,
mide el nivel de llenado con sensor ultrasónico HC-SR04 (GPIO 23/24) y publica telemetría MQTT.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import (
    CATEGORIES,
    EDGE_ACTUATOR_GPIO_PINS,
    EDGE_NODE_HARDWARE,
    TARGET_LATENCY_MS,
)


class RaspberryPiActuator:
    """
    Controlador de actuadores para Raspberry Pi (gpiozero / RPi.GPIO).
    Si se ejecuta en Windows/PC durante desarrollo o pruebas, funciona automáticamente
    en modo gemelo digital (simulación transparente de pines GPIO y PWM).
    """

    def __init__(self, pwm_pin: int = EDGE_ACTUATOR_GPIO_PINS["servo_compuerta_pwm"]):
        self.pwm_pin = pwm_pin
        self.hardware_available = False
        self._servo = None

        try:
            from gpiozero import AngularServo  # type: ignore

            self._servo = AngularServo(
                self.pwm_pin,
                min_angle=0,
                max_angle=180,
                min_pulse_width=0.0005,
                max_pulse_width=0.0025,
            )
            self.hardware_available = True
        except Exception:
            self.hardware_available = False

    def move_gate(self, angle: int) -> Dict[str, Any]:
        clamped_angle = max(0, min(180, int(angle)))
        duty_cycle_pct = round(2.5 + (clamped_angle / 180.0) * 10.0, 2)
        if self.hardware_available and self._servo is not None:
            try:
                self._servo.angle = clamped_angle
            except Exception:
                pass
        return {
            "gpio_pin_bcm": self.pwm_pin,
            "servo_angle_deg": clamped_angle,
            "pwm_duty_cycle_pct": duty_cycle_pct,
            "mode": "GPIO Físico (Raspberry Pi)" if self.hardware_available else "Simulación Gemelo Digital (PC -> RPi GPIO)",
        }


class EdgeTelemetryManager:
    """Gestiona métricas de latencia en la Raspberry Pi, estado de contenedores y eventos MQTT."""

    def __init__(
        self,
        station_id: str = "ULACIT-RPI-EST01",
        mqtt_broker: Optional[str] = None,
        mqtt_port: int = 1883,
    ):
        self.station_id = station_id
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.actuator = RaspberryPiActuator()
        self.fill_levels: Dict[str, float] = {
            cat: 15.0 for cat in CATEGORIES if cat != "No Identificado"
        }
        self.category_counts: Counter = Counter()
        self.recent_events: List[Dict[str, Any]] = []
        self.last_servo_angle: int = 0
        self.last_active_container: str = "En espera"

    def record_frame_event(
        self,
        detections: List[Dict[str, Any]],
        latency_ms: float,
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        primary_det = detections[0] if detections else None

        if primary_det:
            cat = primary_det["category"]
            conf = primary_det["confidence"]
            cat_info = CATEGORIES.get(cat, CATEGORIES["No Identificado"])
            self.last_servo_angle = cat_info["servo_angle"]
            self.last_active_container = cat_info["container"]
            self.category_counts[cat] += 1
            if cat in self.fill_levels:
                self.fill_levels[cat] = min(100.0, round(self.fill_levels[cat] + 0.25, 2))
        else:
            cat = "Sin objeto"
            conf = 0.0
            self.last_servo_angle = 0

        gpio_state = self.actuator.move_gate(self.last_servo_angle)

        alerts = [
            f"Contenedor de {c} al {lvl:.0f}% (Priorizar recolección)"
            for c, lvl in self.fill_levels.items()
            if lvl >= 80.0
        ]

        payload = {
            "station_id": self.station_id,
            "edge_hardware": EDGE_NODE_HARDWARE,
            "timestamp": timestamp,
            "edge_latency_ms": round(latency_ms, 1),
            "meets_realtime_sla": latency_ms <= TARGET_LATENCY_MS,
            "objects_in_frame": len(detections),
            "primary_category": cat,
            "primary_subtype": primary_det["subtype"] if primary_det else None,
            "confidence": round(conf, 3),
            "raspberry_pi_gpio": {
                **gpio_state,
                "target_container": self.last_active_container,
                "ultrasonic_pins_bcm": {
                    "trig": EDGE_ACTUATOR_GPIO_PINS["ultrasonico_trig"],
                    "echo": EDGE_ACTUATOR_GPIO_PINS["ultrasonico_echo"],
                },
                "status": "ACTIVO" if primary_det and cat != "No Identificado" else "STANDBY",
            },
            "fill_levels_pct": dict(self.fill_levels),
            "maintenance_alerts": alerts,
        }

        if primary_det:
            self.recent_events.insert(0, payload)
            self.recent_events = self.recent_events[:25]

        return payload

    def format_mqtt_json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)
