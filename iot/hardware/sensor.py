from __future__ import annotations

import logging
import time

from .base import Sensor
import config

logger = logging.getLogger(__name__)

try:
    from gpiozero import DistanceSensor, MotionSensor
except ImportError:
    DistanceSensor = None
    MotionSensor = None

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class ProximitySensor(Sensor):
    def __init__(self, echo: int = config.PIN_DISTANCE_ECHO, trigger: int = config.PIN_DISTANCE_TRIGGER):
        self._device = None
        if DistanceSensor is not None and not config.MOCK_MODE:
            try:
                self._device = DistanceSensor(echo=echo, trigger=trigger, max_distance=2.0)
            except Exception as exc:
                logger.warning("Distance sensor unavailable: %s", exc)

    def read(self) -> float:
        if self._device is None:
            return 0.35
        try:
            return float(self._device.distance)
        except Exception:
            return 0.35

    def read_cm(self) -> float:
        return round(self.read() * 100.0, 2)

    def close(self) -> None:
        if self._device is not None:
            self._device.close()


class MotionSensorInput(Sensor):
    def __init__(self, pin: int = config.PIN_MOTION):
        self._device = None
        if MotionSensor is not None and not config.MOCK_MODE:
            try:
                self._device = MotionSensor(pin)
            except Exception as exc:
                logger.warning("Motion sensor unavailable: %s", exc)

    def read(self) -> bool:
        if self._device is None:
            return True
        try:
            return bool(self._device.motion_detected)
        except Exception:
            return False

    def close(self) -> None:
        if self._device is not None:
            self._device.close()


class TouchSensor(Sensor):
    def __init__(self, pin: int = config.PIN_TOUCH):
        self.pin = pin
        self.baseline = 0.0
        if GPIO is not None and not config.MOCK_MODE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self.calibrate()

    def _rc_time(self, timeout_s: float = 0.5) -> int:
        if GPIO is None or config.MOCK_MODE:
            return 120

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

    def average(self, samples: int = 10, delay: float = 0.02) -> float:
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

    def is_touched(self, delta_ratio: float = 0.25) -> bool:
        value = self.read()
        baseline = self.baseline or value
        return value > baseline * (1 + delta_ratio)

    def snapshot(self) -> dict:
        value = self.read()
        baseline = self.baseline or value
        return {
            "value": round(value, 2),
            "baseline": round(baseline, 2),
            "is_touched": value > baseline * 1.25,
        }

    def close(self) -> None:
        pass

