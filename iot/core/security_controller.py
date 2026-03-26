from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from biometrics.biometrics_service import build_multimodal_profile
from hardware.buzzer import Buzzer
from hardware.camera import AccessCamera
from hardware.lcd import LCDDisplay
from hardware.led import StatusLED
from hardware.sensor import LightSensor
import config

logger = logging.getLogger(__name__)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class SecurityController:
    def __init__(self):
        self.green_led = StatusLED(config.PIN_LED_GREEN, name="Green LED")
        self.red_led = StatusLED(config.PIN_LED_RED, name="Red LED")
        self.light_led_1 = StatusLED(config.PIN_LIGHT_LED_1, name="Light LED 1")
        self.light_led_2 = StatusLED(config.PIN_LIGHT_LED_2, name="Light LED 2")
        self.buzzer = Buzzer(config.PIN_BUZZER)
        self.lcd = LCDDisplay()
        self.light_sensor = LightSensor(config.PIN_LIGHT_SENSOR)
        self.camera = AccessCamera()
        self.auto_light_enabled = True
        self.manual_light_enabled = False

        self.boot_sequence()

    def boot_sequence(self) -> None:
        self.lcd.show_message(config.APP_SHORT_NAME, "Init capteurs")
        self.green_led.on()
        self.red_led.on()
        self.buzzer.beep(count=2, on_time=0.15, off_time=0.15)
        time.sleep(0.3)
        self.green_led.off()
        self.red_led.off()
        self.reset_idle()

    def reset_idle(self) -> None:
        self.sync_lighting()
        if self.auto_light_enabled:
            self.lcd.show_message(config.APP_SHORT_NAME, "Scan pret")
        else:
            self.lcd.show_message(config.APP_SHORT_NAME, "Mode manuel")

    def handle_scanning(self) -> None:
        self.sync_lighting(force_on=True)
        self.lcd.show_message("Analyse en cours", "Ne bouge pas")

    def handle_enrollment(self, user_id: str) -> None:
        self.sync_lighting(force_on=True)
        self.lcd.show_message("Enrolement", user_id[: config.LCD_COLS])

    def handle_access_granted(self, username: str = "UTILISATEUR", score: float | None = None) -> None:
        label = username[: config.LCD_COLS]
        line2 = f"Score {score:.2f}"[: config.LCD_COLS] if score is not None else "Bienvenue"
        self.lcd.show_message("Acces autorise", line2)
        self.green_led.on()
        self.red_led.off()
        self.buzzer.beep(count=1, on_time=0.3, off_time=0.1)
        time.sleep(1.5)
        self.green_led.off()
        self.reset_idle()
        logger.info("Access granted for %s", label)

    def handle_access_denied(self, reason: str = "Profil inconnu") -> None:
        self.lcd.show_message("Acces refuse", reason[: config.LCD_COLS])
        self.red_led.blink(count=3, period=0.12)
        self.buzzer.beep(count=3, on_time=0.1, off_time=0.1)
        time.sleep(0.5)
        self.reset_idle()

    def _assist_lights_on(self) -> None:
        self.light_led_1.on()
        self.light_led_2.on()

    def _assist_lights_off(self) -> None:
        self.light_led_1.off()
        self.light_led_2.off()

    def sync_lighting(self, force_on: bool = False) -> dict:
        light = self.light_sensor.snapshot()
        should_enable = force_on or self.manual_light_enabled
        if self.auto_light_enabled and light["is_dark"]:
            should_enable = True

        if should_enable:
            self._assist_lights_on()
        else:
            self._assist_lights_off()

        return {
            "auto_light_enabled": self.auto_light_enabled,
            "manual_light_enabled": self.manual_light_enabled,
            "assist_lights_on": self.light_led_1.state and self.light_led_2.state,
            "ambient": light,
        }

    def sensor_snapshot(self, include_preview: bool = False) -> dict:
        lighting = self.sync_lighting()
        camera_snapshot = self.camera.snapshot()
        if include_preview:
            camera_snapshot["preview_jpeg_base64"] = self.camera.capture_preview_base64(
                width=config.CAMERA_PREVIEW_WIDTH,
                quality=config.CAMERA_PREVIEW_QUALITY,
            )
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "device_id": config.DEVICE_ID,
            "light_sensor": lighting["ambient"],
            "lighting": {
                "auto_enabled": lighting["auto_light_enabled"],
                "manual_enabled": lighting["manual_light_enabled"],
                "assist_lights_on": lighting["assist_lights_on"],
                "green_led_on": self.green_led.state,
                "red_led_on": self.red_led.state,
            },
            "buzzer": self.buzzer.snapshot(),
            "lcd": self.lcd.snapshot(),
            "camera": camera_snapshot,
        }

    def capture_attempt(self, claimed_user_id: str | None = None, persist_preview: bool = True) -> dict:
        self.handle_scanning()
        frame = self.camera.capture_array()
        telemetry = self.sensor_snapshot()
        preview_path = None

        if persist_preview:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_path = Path(config.CAPTURE_DIR) / f"{claimed_user_id or 'scan'}_{timestamp}.jpg"
            self.camera.capture_to_file(preview_path)

        # Precompute profile to surface segmentation failures early.
        profile = build_multimodal_profile(frame)

        return {
            "frame": frame,
            "telemetry": telemetry,
            "preview_path": str(preview_path) if preview_path else None,
            "profile": profile,
        }

    def apply_remote_settings(self, payload: dict) -> dict:
        if "auto_light_enabled" in payload:
            self.auto_light_enabled = _coerce_bool(payload["auto_light_enabled"])

        if "assist_lights_on" in payload:
            self.manual_light_enabled = _coerce_bool(payload["assist_lights_on"])

        if "dark_ratio" in payload:
            try:
                self.light_sensor.set_dark_ratio(float(payload["dark_ratio"]))
            except (TypeError, ValueError):
                logger.warning("Invalid dark_ratio received: %s", payload.get("dark_ratio"))

        if "green_led_on" in payload:
            self.green_led.set_state(_coerce_bool(payload["green_led_on"]))

        if "red_led_on" in payload:
            self.red_led.set_state(_coerce_bool(payload["red_led_on"]))

        if payload.get("buzzer_test"):
            self.buzzer.beep(count=1, on_time=0.2, off_time=0.1)

        if "lcd_line1" in payload or "lcd_line2" in payload:
            line1 = str(payload.get("lcd_line1", "")).strip()
            line2 = str(payload.get("lcd_line2", "")).strip()
            if line1 or line2:
                self.lcd.show_message(line1 or config.APP_SHORT_NAME, line2)
            else:
                self.reset_idle()

        telemetry = self.sensor_snapshot()
        return {
            "auto_light_enabled": self.auto_light_enabled,
            "assist_lights_on": telemetry["lighting"]["assist_lights_on"],
            "green_led_on": telemetry["lighting"]["green_led_on"],
            "red_led_on": telemetry["lighting"]["red_led_on"],
            "dark_ratio": telemetry["light_sensor"]["dark_ratio"],
            "lcd": telemetry["lcd"],
            "buzzer": telemetry["buzzer"],
            "telemetry": telemetry,
        }

    def close(self) -> None:
        for device in (
            self.green_led,
            self.red_led,
            self.light_led_1,
            self.light_led_2,
            self.buzzer,
            self.lcd,
            self.light_sensor,
            self.camera,
        ):
            try:
                device.close()
            except Exception as exc:
                logger.warning("Device close failed: %s", exc)
