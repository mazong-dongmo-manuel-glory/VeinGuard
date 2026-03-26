from __future__ import annotations

import base64
import logging
import time
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

try:
    from libcamera import controls as libcamera_controls
except ImportError:
    libcamera_controls = None


class AccessCamera(Camera):
    def __init__(self):
        self._camera = None
        self.available = False
        if Picamera2 is not None and not config.MOCK_MODE:
            try:
                self._camera = Picamera2()
                preview = self._camera.create_video_configuration(
                    main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                    controls=self._build_noir_controls(),
                )
                self._camera.configure(preview)
                self._camera.start()
                time.sleep(max(config.NOIR_CAMERA_WARMUP_SECONDS, 0.0))
                self._apply_noir_controls()
                self.available = True
            except Exception as exc:
                logger.warning("Camera unavailable, fallback to mock frame: %s", exc)
                self._camera = None

    def _build_noir_controls(self) -> dict:
        controls = {
            "AeEnable": True,
            "AwbEnable": False,
            "Saturation": 0.0,
            "Brightness": float(config.NOIR_CAMERA_BRIGHTNESS),
            "Contrast": float(config.NOIR_CAMERA_CONTRAST),
            "Sharpness": float(config.NOIR_CAMERA_SHARPNESS),
            "ExposureValue": float(config.NOIR_CAMERA_EXPOSURE_VALUE),
            "FrameDurationLimits": (
                int(config.CAMERA_FRAME_DURATION_US),
                int(config.CAMERA_FRAME_DURATION_US),
            ),
        }
        if libcamera_controls is not None:
            try:
                controls["NoiseReductionMode"] = libcamera_controls.draft.NoiseReductionModeEnum.HighQuality
            except Exception:
                pass
        return controls

    def _apply_noir_controls(self) -> None:
        if self._camera is None:
            return
        try:
            # The NoIR module is used as a near-IR texture sensor, so we favor stable exposure and local contrast.
            self._camera.set_controls(self._build_noir_controls())
        except Exception as exc:
            logger.debug("Unable to apply NoIR controls: %s", exc)

    def capture_array(self):
        if self._camera is not None:
            frame_rgb = self._camera.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self._mock_frame()

    def capture_bytes(self) -> bytes:
        frame = self.capture_array()
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else b""

    def capture_preview_base64(self, width: int = 320, quality: int = 60) -> str:
        frame = self.capture_array()
        if width and frame.shape[1] > width:
            ratio = width / float(frame.shape[1])
            height = int(frame.shape[0] * ratio)
            frame = cv2.resize(frame, (width, height))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("ascii")

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
            "frame_duration_us": config.CAMERA_FRAME_DURATION_US,
            "contrast": config.NOIR_CAMERA_CONTRAST,
            "sharpness": config.NOIR_CAMERA_SHARPNESS,
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
