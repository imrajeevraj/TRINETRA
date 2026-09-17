import threading
import time
import logging
import serial
import pynmea2
import asyncio
from backend.app.api.ws import manager

logger = logging.getLogger("GeoService")


class GeoService:
    def __init__(self):
        self.running = False
        self.thread = None
        self.loop = None

        # Configuration for GPS hardware. Could be moved to settings.
        self.port = "COM3"  # e.g. /dev/ttyUSB0 or COM3
        self.baudrate = 9600

        # Current command post location (simulated default if hardware fails)
        self.current_lat = 28.6139
        self.current_lon = 77.2090
        self.simulation_mode = True

    def start(self, loop=None):
        if self.running:
            return
        self.loop = loop or asyncio.get_event_loop()
        self.running = True
        self.thread = threading.Thread(target=self._gps_loop, daemon=True)
        self.thread.start()
        logger.info("GeoService started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("GeoService stopped")

    def _gps_loop(self):
        ser = None
        try:
            ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
            self.simulation_mode = False
            logger.info(f"Connected to GPS device on {self.port}")
        except Exception as e:
            logger.warning(
                f"Could not connect to GPS on {self.port}: {e}. Falling back to simulation mode."
            )
            self.simulation_mode = True

        sim_direction_lat = 0.00001
        sim_direction_lon = 0.00001

        while self.running:
            try:
                if not self.simulation_mode and ser and ser.is_open:
                    line = ser.readline().decode("ascii", errors="replace")
                    if line.startswith("$GPGGA") or line.startswith("$GPRMC"):
                        try:
                            msg = pynmea2.parse(line)
                            if hasattr(msg, "latitude") and hasattr(msg, "longitude"):
                                self.current_lat = msg.latitude
                                self.current_lon = msg.longitude
                                self._broadcast_location()
                        except pynmea2.ParseError:
                            pass
                else:
                    # Simulation mode: wander slightly around the default location
                    self.current_lat += sim_direction_lat
                    self.current_lon += sim_direction_lon

                    if self.current_lat > 28.6150 or self.current_lat < 28.6120:
                        sim_direction_lat *= -1
                    if self.current_lon > 77.2110 or self.current_lon < 77.2070:
                        sim_direction_lon *= -1

                    self._broadcast_location()
                    time.sleep(2.0)
            except Exception as e:
                logger.error(f"Error in GPS loop: {e}")
                time.sleep(2.0)

        if ser and ser.is_open:
            ser.close()

    def _broadcast_location(self):
        if not self.loop or self.loop.is_closed():
            return

        payload = {
            "event_type": "gps_update",
            "data": {
                "bop_location": {"lat": self.current_lat, "lng": self.current_lon}
            },
        }

        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(payload), self.loop)
        except Exception:
            pass


geo_service = GeoService()
