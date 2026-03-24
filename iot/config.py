# Configuration for VeinGuard IoT Backend

# MQTT Settings
MQTT_BROKER = "localhost" # Or PC IP if using external broker
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60
MQTT_TOPIC_CMD = "veinguard/cmd/#"
MQTT_TOPIC_RES = "veinguard/res/#"
MQTT_TOPIC_STATUS = "veinguard/status"
MQTT_TOPIC_LOGS = "veinguard/logs"

# Hardware Settings (GPIO BCM Pins)
PIN_LED_GREEN = 17
PIN_LED_RED = 27
PIN_SENSOR_TRIGGER = 23
PIN_SENSOR_ECHO = 24

# Database Settings
DB_PATH = "veinguard.db"
