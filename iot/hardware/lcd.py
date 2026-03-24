from .base import Display

try:
    # Assuming a common I2C LCD library like RPLCD or similar
    # If the user has a specific one, they can adjust here.
    # For now, we provide a generic structure.
    import smbus
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False

class LCDDisplay(Display):
    """
    Controls an LCD (e.g., 16x2 I2C).
    Falls back to Console Mock if I2C/Library is not available.
    """
    def __init__(self):
        if I2C_AVAILABLE:
            # Placeholder for real I2C LCD initialization
            # from RPLCD.i2c import CharLCD
            # self.lcd = CharLCD('PCF8574', 0x27)
            pass
        else:
            print("[HAL] LCD initialized (MOCK MODE)")

    def show_message(self, line1, line2=""):
        if I2C_AVAILABLE:
            # self.lcd.clear()
            # self.lcd.write_string(f"{line1}\n{line2}")
            pass
        else:
            print(f"\n[LCD DISPLAY]")
            print(f"| {line1.center(16)} |")
            print(f"| {line2.center(16)} |")
            print("-" * 20)

    def clear(self):
        if I2C_AVAILABLE:
            # self.lcd.clear()
            pass
        else:
            print("[HAL] LCD Cleared")

if __name__ == "__main__":
    lcd = LCDDisplay()
    lcd.show_message("VeinGuard v1.0", "SYSTEM READY")
