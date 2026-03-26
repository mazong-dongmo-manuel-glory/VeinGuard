from __future__ import annotations

from abc import ABC, abstractmethod


class Closable(ABC):
    def close(self) -> None:
        pass


class Actuator(Closable, ABC):
    @abstractmethod
    def on(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def off(self) -> None:
        raise NotImplementedError


class Display(Closable, ABC):
    @abstractmethod
    def show_message(self, line1: str, line2: str = "") -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class Camera(Closable, ABC):
    @abstractmethod
    def capture_array(self):
        raise NotImplementedError


class Sensor(Closable, ABC):
    @abstractmethod
    def read(self):
        raise NotImplementedError

