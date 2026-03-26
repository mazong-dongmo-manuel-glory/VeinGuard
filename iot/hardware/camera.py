from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import Camera
import config

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class AccessCamera(Camera):
    def __init__(self):
        self._camera = None
        self.available = False
        if Picamera2 is not None and not config.MOCK_MODE:
            try:
                self._camera = Picamera2()
                preview = self._camera.create_preview_configuration(
                    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"}
                )
                self._camera.configure(preview)
                self._camera.start()
                self.available = True
            except Exception as exc:
                logger.warning("Camera unavailable, fallback to mock frame: %s", exc)
                self._camera = None

    def capture_array(self):
        if self._camera is not None:
            frame_rgb = self._camera.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self._mock_frame()

    def capture_bytes(self) -> bytes:
        frame = self.capture_array()
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else b""

    def capture_to_file(self, file_path: str | Path) -> str:
        file_path = str(file_path)
        cv2.imwrite(file_path, self.capture_array())
        return file_path

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()

    def snapshot(self) -> dict:
        return {
            "available": self.available or config.MOCK_MODE,
            "mock_mode": config.MOCK_MODE or not self.available,
            "width": config.CAMERA_WIDTH,
            "height": config.CAMERA_HEIGHT,
        }

    def _mock_frame(self):
        frame = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), 235, dtype=np.uint8)
        cv2.rectangle(frame, (170, 80), (470, 420), (190, 170, 150), -1)
        cv2.circle(frame, (250, 120), 40, (160, 140, 120), -1)
        cv2.circle(frame, (320, 95), 36, (160, 140, 120), -1)
        cv2.circle(frame, (390, 100), 34, (160, 140, 120), -1)
        cv2.circle(frame, (450, 130), 30, (160, 140, 120), -1)
        cv2.putText(frame, "MOCK PALM", (210, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 80, 90), 2)
        return frame
