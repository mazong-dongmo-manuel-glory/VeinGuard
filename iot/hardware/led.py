from __future__ import annotations

import logging
import time

from .base import Actuator

logger = logging.getLogger(__name__)

try:
    from gpiozero import LED as GpioZeroLED
except ImportError:
    GpioZeroLED = None


class StatusLED(Actuator):
    def __init__(self, pin: int, name: str = "LED"):
        self.pin = pin
        self.name = name
        self.state = False
        self._device = None

        if GpioZeroLED is not None:
            try:
                self._device = GpioZeroLED(pin)
            except Exception as exc:
                logger.warning("LED %s unavailable on pin %s: %s", name, pin, exc)

    def on(self) -> None:
        self.state = True
        if self._device is not None:
            self._device.on()
        else:
            logger.info("[MOCK][%s] ON", self.name)

    def off(self) -> None:
        self.state = False
        if self._device is not None:
            self._device.off()
        else:
            logger.info("[MOCK][%s] OFF", self.name)

    def set_state(self, enabled: bool) -> None:
        if enabled:
            self.on()
        else:
            self.off()

    def blink(self, count: int = 3, period: float = 0.2) -> None:
        for _ in range(count):
            self.on()
            time.sleep(period)
            self.off()
            time.sleep(period)

    def snapshot(self) -> dict:
        return {"pin": self.pin, "name": self.name, "is_on": self.state}

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
