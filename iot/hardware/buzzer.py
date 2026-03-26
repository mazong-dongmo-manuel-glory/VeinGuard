from __future__ import annotations

import logging
import time

from .base import Actuator

logger = logging.getLogger(__name__)

try:
    from gpiozero import Buzzer as GpioZeroBuzzer
except ImportError:
    GpioZeroBuzzer = None


class Buzzer(Actuator):
    def __init__(self, pin: int):
        self.pin = pin
        self._device = None
        self.last_action = "OFF"
        if GpioZeroBuzzer is not None:
            try:
                self._device = GpioZeroBuzzer(pin)
            except Exception as exc:
                logger.warning("Buzzer unavailable on pin %s: %s", pin, exc)

    def on(self) -> None:
        self.last_action = "ON"
        if self._device is not None:
            self._device.on()
        else:
            logger.info("[MOCK][BUZZER] ON")

    def off(self) -> None:
        self.last_action = "OFF"
        if self._device is not None:
            self._device.off()
        else:
            logger.info("[MOCK][BUZZER] OFF")

    def beep(self, count: int = 3, on_time: float = 0.3, off_time: float = 0.3) -> None:
        self.last_action = f"BEEP:{count}"
        for _ in range(count):
            self.on()
            time.sleep(on_time)
            self.off()
            time.sleep(off_time)

    def snapshot(self) -> dict:
        return {"pin": self.pin, "last_action": self.last_action}

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
