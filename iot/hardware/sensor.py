from .base import Sensor
import time

try:
    from gpiozero import DistanceSensor
    HAS_GPIOZERO = True
except ImportError:
    HAS_GPIOZERO = False

class VeinDistanceSensor(Sensor):
    def __init__(self, echo=24, trigger=23):
        if HAS_GPIOZERO:
            try:
                self.sensor = DistanceSensor(echo=echo, trigger=trigger)
                print(f"[Sensor] DistanceSensor initialized on Echo:{echo}, Trigger:{trigger}")
            except Exception as e:
                print(f"[Sensor] Failed to init GPIO DistanceSensor: {e}")
                self.sensor = None
        else:
            self.sensor = None
            print("[Sensor] gpiozero not found, using MockDistanceSensor")

    def read(self):
        """Returns distance in meters. Returns 9.9 if sensor is unavailable."""
        if self.sensor:
            try:
                return self.sensor.distance
            except Exception:
                return 9.9
        else:
            # Mock or unavailable
            return 9.9 

if __name__ == "__main__":
    # Test
    sensor = VeinDistanceSensor()
    while True:
        print(f"Distance: {sensor.read():.2f} m")
        time.sleep(1)
