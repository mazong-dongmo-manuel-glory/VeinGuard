from __future__ import annotations

import logging
import time

from .base import Sensor
import config

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class LightSensor(Sensor):
    def __init__(self, pin: int = config.PIN_LIGHT_SENSOR):
        self.pin = pin
        self.baseline = 0.0
        self.dark_ratio = config.LIGHT_SENSOR_DARK_RATIO
        if GPIO is not None and not config.MOCK_MODE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.calibrate()

    def _rc_time(self, timeout_s: float = config.LIGHT_SENSOR_TIMEOUT) -> int:
        if GPIO is None or config.MOCK_MODE:
            return 180

        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.LOW)
        time.sleep(0.02)

        GPIO.setup(self.pin, GPIO.IN)
        count = 0
        start = time.time()
        while GPIO.input(self.pin) == GPIO.LOW:
            count += 1
            if time.time() - start > timeout_s:
                break
        return count

    def average(self, samples: int = config.LIGHT_SENSOR_SAMPLES, delay: float = 0.02) -> float:
        total = 0
        for _ in range(samples):
            total += self._rc_time()
            time.sleep(delay)
        return total / max(samples, 1)

    def calibrate(self) -> float:
        self.baseline = self.average(samples=10, delay=0.01)
        return self.baseline

    def read(self) -> float:
        if GPIO is None or config.MOCK_MODE:
            return 180.0
        return self.average(samples=4, delay=0.01)

    def is_dark(self, dark_ratio: float | None = None) -> bool:
        ratio = dark_ratio if dark_ratio is not None else self.dark_ratio
        value = self.read()
        baseline = self.baseline or value
        return value > baseline * ratio

    def set_dark_ratio(self, ratio: float) -> None:
        self.dark_ratio = max(ratio, 1.01)

    def snapshot(self) -> dict:
        value = self.read()
        baseline = self.baseline or value
        threshold = baseline * self.dark_ratio
        return {
            "value": round(value, 2),
            "baseline": round(baseline, 2),
            "dark_ratio": round(self.dark_ratio, 3),
            "dark_threshold": round(threshold, 2),
            "is_dark": value > threshold,
        }

    def close(self) -> None:
        if GPIO is not None and not config.MOCK_MODE:
            try:
                GPIO.cleanup(self.pin)
            except Exception:
                pass
