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
from hardware.sensor import MotionSensorInput, ProximitySensor, TouchSensor
import config

logger = logging.getLogger(__name__)


class SecurityController:
    def __init__(self):
        self.green_led = StatusLED(config.PIN_LED_GREEN, name="Green LED")
        self.red_led = StatusLED(config.PIN_LED_RED, name="Red LED")
        self.buzzer = Buzzer(config.PIN_BUZZER)
        self.lcd = LCDDisplay()
        self.touch_sensor = TouchSensor(config.PIN_TOUCH)
        self.proximity_sensor = ProximitySensor(
            echo=config.PIN_DISTANCE_ECHO,
            trigger=config.PIN_DISTANCE_TRIGGER,
        )
        self.motion_sensor = MotionSensorInput(config.PIN_MOTION)
        self.camera = AccessCamera()

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
        self.lcd.show_message(config.APP_SHORT_NAME, "Main / doigt")

    def handle_scanning(self) -> None:
        self.lcd.show_message("Analyse en cours", "Ne bouge pas")

    def handle_enrollment(self, user_id: str) -> None:
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

    def sensor_snapshot(self) -> dict:
        touch_state = self.touch_sensor.snapshot()
        return {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "device_id": config.DEVICE_ID,
            "distance_cm": self.proximity_sensor.read_cm(),
            "motion_detected": self.motion_sensor.read(),
            "touch": touch_state,
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

    def close(self) -> None:
        for device in (
            self.green_led,
            self.red_led,
            self.buzzer,
            self.lcd,
            self.touch_sensor,
            self.proximity_sensor,
            self.motion_sensor,
            self.camera,
        ):
            try:
                device.close()
            except Exception as exc:
                logger.warning("Device close failed: %s", exc)
