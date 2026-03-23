import os
from .base import Camera

try:
    from picamzero import Camera as PiCam
    HAS_PICAM = True
except ImportError:
    HAS_PICAM = False

class VeinCamera(Camera):
    def __init__(self):
        if HAS_PICAM:
            self.cam = PiCam()
            print("[Camera] PiCamera initialized via picamzero")
        else:
            print("[Camera] PiCamera not found, using MockCamera")

    def capture(self, filename):
        if HAS_PICAM:
            self.cam.take_photo(filename)
            print(f"[Camera] Photo captured: {filename}")
        else:
            print(f"[Camera] MOCK: Capturing vein image to {filename}")
            # In a real mock, we might copy a sample image
            with open(filename, 'w') as f:
                f.write("MOCK_IMAGE_DATA")

if __name__ == "__main__":
    # Test
    cam = VeinCamera()
    cam.capture("test_vein.jpg")
