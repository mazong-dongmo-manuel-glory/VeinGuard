from .base import Actuator
import time


try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

class StatusLED(Actuator):
    """
    Controls a status LED (Green or Red).
    Falls back to Console Mock if RPi.GPIO is not available.
    """
    def __init__(self, pin, name="LED"):
        self.pin = pin
        self.name = name
        self.state = False
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HAL] {self.name} initialized on Pin {self.pin} (MOCK MODE)")

    def on(self):
        self.state = True
        if GPIO_AVAILABLE:
            GPIO.output(self.pin, GPIO.HIGH)
        else:
            print(f"[HAL] {self.name} is now ON")

    def off(self):
        self.state = False
        if GPIO_AVAILABLE:
            GPIO.output(self.pin, GPIO.LOW)
        else:
            print(f"[HAL] {self.name} is now OFF")

    def blink(self, count=3, period=0.2):
        for _ in range(count):
            self.on()
            time.sleep(period)
            self.off()
            time.sleep(period)

if __name__ == "__main__":
    # Test script
    green = StatusLED(17, "Green LED")
    red = StatusLED(27, "Red LED")
    
    print("Testing Green LED...")
    green.blink()
    print("Testing Red LED...")
    red.blink()
