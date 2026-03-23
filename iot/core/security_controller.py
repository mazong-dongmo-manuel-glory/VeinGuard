from hardware.led import StatusLED
from hardware.lcd import LCDDisplay
from hardware.sensor import VeinDistanceSensor
import config

class SecurityController:
    """
    Orchestrates the security hardware and access logic.
    Provides a high-level interface for the API to trigger hardware actions.
    """
    def __init__(self):
        # Hardware Initialization
        self.green_led = StatusLED(pin=config.PIN_LED_GREEN, name="Green LED")
        self.red_led = StatusLED(pin=config.PIN_LED_RED, name="Red LED")
        self.lcd = LCDDisplay()
        self.sensor = VeinDistanceSensor(
            echo=config.PIN_SENSOR_ECHO, 
            trigger=config.PIN_SENSOR_TRIGGER
        )
        
        self.boot_sequence()

    def check_proximity(self, threshold=0.1):
        """Returns True if an object is within threshold meters."""
        dist = self.sensor.read()
        return dist < threshold

    def boot_sequence(self):
        """Initial feedback when the system starts."""
        self.lcd.show_message("VeinGuard v1.0", "SYSTEM INITIALIZING")
        self.green_led.on()
        self.red_led.on()
        time.sleep(1)
        self.green_led.off()
        self.red_led.off()
        self.lcd.show_message("VeinGuard", "READY FOR SCAN")

    def handle_access_granted(self, username="USER"):
        """Sequence for authorized access."""
        self.lcd.show_message("ACCESS GRANTED", f"WELCOME {username.upper()}")
        self.green_led.on()
        time.sleep(2)
        self.green_led.off()
        self.reset_idle()

    def handle_access_denied(self, reason="VASCULAR MISMATCH"):
        """Sequence for unauthorized access."""
        self.lcd.show_message("ACCESS DENIED", reason)
        self.red_led.blink(count=3, period=0.1)
        time.sleep(1)
        self.reset_idle()

    def handle_scanning(self):
        """Visual feedback during biometric capture."""
        self.lcd.show_message("SCANNING...", "DO NOT MOVE")

    def reset_idle(self):
        """Returns the hardware to idle state."""
        self.lcd.show_message("VeinGuard", "READY FOR SCAN")

if __name__ == "__main__":
    # Test the controller
    ctrl = SecurityController()
    time.sleep(1)
    ctrl.handle_scanning()
    time.sleep(2)
    ctrl.handle_access_granted("ADMIN")
    time.sleep(2)
    ctrl.handle_access_denied()
