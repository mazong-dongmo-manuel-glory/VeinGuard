from abc import ABC, abstractmethod

class Actuator(ABC):
    """Base class for all actuators (LEDs, Relays, etc.)"""
    @abstractmethod
    def on(self):
        pass

    @abstractmethod
    def off(self):
        pass

class Display(ABC):
    """Base class for visual feedback (LCD, OLED, etc.)"""
    @abstractmethod
    def show_message(self, line1, line2=""):
        pass

    @abstractmethod
    def clear(self):
        pass

class Camera(ABC):
    """Base class for image capture (PiCamera, Webcam, etc.)"""
    @abstractmethod
    def capture(self, filename):
        pass

class Sensor(ABC):
    """Base class for environmental sensors (Distance, PIR, etc.)"""
    @abstractmethod
    def read(self):
        pass
