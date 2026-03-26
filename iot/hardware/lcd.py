from __future__ import annotations

import logging
import time

from .base import Display
import config

logger = logging.getLogger(__name__)

try:
    from RPLCD.i2c import CharLCD
except ImportError:
    CharLCD = None


class LCDDisplay(Display):
    def __init__(self, address: int = config.LCD_I2C_ADDRESS, port: int = config.LCD_PORT):
        self.address = address
        self.port = port
        self._lcd = None
        self.last_message = {"line1": "", "line2": ""}

        if not config.MOCK_MODE and CharLCD is not None:
            self._init_lcd()

    def _init_lcd(self) -> None:
        candidates = [self.address, 0x3F if self.address != 0x3F else 0x27]
        for addr in candidates:
            try:
                self._lcd = CharLCD(
                    i2c_expander="PCF8574",
                    address=addr,
                    port=self.port,
                    cols=config.LCD_COLS,
                    rows=config.LCD_ROWS,
                    dotsize=8,
                    auto_linebreaks=True,
                    backlight_enabled=True,
                )
                self._lcd.clear()
                self.address = addr
                logger.info("LCD initialised at %s", hex(addr))
                return
            except Exception as exc:
                logger.warning("LCD init failed at %s: %s", hex(addr), exc)
        self._lcd = None

    def show_message(self, line1: str, line2: str = "") -> None:
        line1 = self._format(line1)
        line2 = self._format(line2)
        self.last_message = {"line1": line1.strip(), "line2": line2.strip()}
        logger.info("[LCD] |%s| |%s|", line1, line2)
        if self._lcd is not None:
            self._write(line1, line2)

    def scroll(self, text: str, delay: float = 1.2) -> None:
        words = text.split()
        line1 = ""
        line2 = ""

        def flush() -> None:
            self.show_message(line1, line2)
            time.sleep(delay)

        for word in words:
            candidate1 = f"{line1} {word}".strip()
            if len(candidate1) <= config.LCD_COLS:
                line1 = candidate1
                continue

            candidate2 = f"{line2} {word}".strip()
            if len(candidate2) <= config.LCD_COLS:
                line2 = candidate2
                continue

            flush()
            line1 = word
            line2 = ""

        if line1 or line2:
            flush()
        self.clear()

    def clear(self) -> None:
        self.last_message = {"line1": "", "line2": ""}
        if self._lcd is not None:
            self._lcd.clear()

    def backlight(self, enabled: bool) -> None:
        if self._lcd is not None:
            self._lcd.backlight_enabled = enabled

    def close(self) -> None:
        if self._lcd is not None:
            try:
                self._lcd.clear()
                self._lcd.close(clear=True)
            except Exception as exc:
                logger.warning("LCD close failed: %s", exc)

    def _format(self, text: str) -> str:
        return f"{str(text)[:config.LCD_COLS]:<{config.LCD_COLS}}"

    def _write(self, line1: str, line2: str) -> None:
        try:
            self._lcd.clear()
            self._lcd.home()
            self._lcd.write_string(line1[: config.LCD_COLS])
            self._lcd.cursor_pos = (1, 0)
            self._lcd.write_string(line2[: config.LCD_COLS])
        except Exception as exc:
            logger.warning("LCD write failed: %s", exc)
            self._init_lcd()

    def snapshot(self) -> dict:
        return {
            "address": hex(self.address),
            "line1": self.last_message["line1"],
            "line2": self.last_message["line2"],
            "enabled": self._lcd is not None or config.MOCK_MODE,
        }
