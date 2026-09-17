import os
import yaml
import threading
import logging
from typing import Dict, List, Any
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("Config")


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://ibvap_user:ibvap_pass@localhost:5432/ibvap_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    # I-02: Redis password — must match requirepass in Redis container config.
    REDIS_PASSWORD: str = ""
    JWT_SECRET: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ADMIN_USERNAME: str
    ADMIN_PASSWORD: str
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
    MODEL_PATH: str = "models/current/ibvap_detector.pt"
    MODEL_SHA256: str = ""
    MODEL_REGISTRY_PATH: str = "models/model_registry.yaml"
    SPECIALIZED_THREAT_MODEL_PATH: str = "custom_threats.pt"
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"  # nosec B104
    # Demo accounts are opt-in so production deployments never create a
    # known credential by accident.
    DEMO_MODE: bool = False
    DEMO_USERNAME: str = "demo_operator"
    DEMO_PASSWORD: str = ""
    COOKIE_SECURE: bool = True

    # Performance tuning
    AI_INFERENCE_INTERVAL: int = 2

    # External Alert Dispatch
    MQTT_ENABLED: bool = False
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_TOPIC_PREFIX: str = "ibvap/alerts"
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    # S-09: TLS for MQTT transport encryption.
    MQTT_TLS_ENABLED: bool = False
    MQTT_CA_CERT: str = ""  # Path to CA cert bundle; empty = system CA

    TELEGRAM_ENABLED: bool = False
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    model_config = SettingsConfigDict(
        env_file=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../.env")
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Path to YAML configs directory
CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../configs")
)


def load_yaml_config(filename: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}


def get_cameras_config() -> List[Dict[str, Any]]:
    config = load_yaml_config("cameras.yaml")
    return config.get("cameras", [])


def get_risk_config() -> Dict[str, Any]:
    config = load_yaml_config("risk.yaml")
    return config.get("risk_rules", {})


def get_zones_config() -> Dict[str, Any]:
    config = load_yaml_config("zones.yaml")
    return config.get("zones", {})


def get_system_config() -> Dict[str, Any]:
    config = load_yaml_config("system.yaml")
    return config.get("system", {})


# ============================================================================
# ConfigManager: Hot-reload capable configuration manager
# ============================================================================


class ConfigManager:
    """Singleton configuration manager with file watching and hot-reload.

    This class watches YAML config files and automatically reloads them
    if they change, without requiring an application restart.

    Usage:
        from backend.app.core.config import config_manager

        # Get entire config
        risk_config = config_manager.get_config("risk")

        # Get specific value with dot notation
        person_weight = config_manager.get_value("risk", "weights.person", 20)
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if ConfigManager._initialized:
            return

        with self._lock:
            if ConfigManager._initialized:
                return

            self.config_dir = Path(CONFIG_DIR)
            self.configs: Dict[str, Any] = {}
            self.file_times: Dict[str, float] = {}
            self.subscribers: Dict[str, list] = {}
            self._file_lock = threading.Lock()
            # R-03: Stop event allows the watcher thread to exit gracefully.
            self._stop_event = threading.Event()

            # Load initial configs
            self._load_all_configs()
            ConfigManager._initialized = True

            # Start file watcher thread
            self._start_file_watcher()
            logger.info(
                "Config manager initialized with %d configuration files",
                len(self.configs),
            )

    def _load_all_configs(self):
        """Load all YAML config files."""
        config_files = {
            "risk": "risk.yaml",
            "anpr": "anpr.yaml",
            "system": "system.yaml",
            "zones": "zones.yaml",
            "watchlist": "watchlist.yaml",
            "cameras": "cameras.yaml",
        }

        for key, filename in config_files.items():
            filepath = self.config_dir / filename
            if filepath.exists():
                self._load_config_file(key, filepath)
            else:
                logger.warning("Configuration file not found: %s", filename)

    def _load_config_file(self, key: str, filepath: Path):
        """Load a single YAML config file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                config = {}

            with self._file_lock:
                self.configs[key] = config
                self.file_times[key] = os.path.getmtime(filepath)

            logger.info("Loaded configuration: %s", key)
        except Exception as e:
            logger.exception("Unable to load configuration %s: %s", key, e)

    def _start_file_watcher(self):
        """Start background thread to watch config files for changes."""

        def watch_files():
            # R-03: Check stop event on each iteration so the thread exits cleanly.
            while not self._stop_event.wait(timeout=5):
                try:
                    self._check_config_updates()
                except Exception as e:
                    logger.exception("Configuration watcher error: %s", e)

        watcher_thread = threading.Thread(
            target=watch_files, daemon=True, name="ConfigFileWatcher"
        )
        watcher_thread.start()
        logger.info("Configuration file watcher started (checks every 5s)")

    def stop_watching(self) -> None:
        """R-03: Signal the file-watcher thread to exit on its next iteration."""
        self._stop_event.set()
        logger.info("Configuration file watcher stopped")

    @classmethod
    def reset_for_testing(cls) -> None:
        """C-04: Destroy the ConfigManager singleton and all loaded configs.

        Call this in test setUp/teardown to guarantee a pristine state.
        After calling this, the next access to `config_manager` will create
        a fresh instance.

        Example::

            def tearDown(self):
                ConfigManager.reset_for_testing()
        """
        with cls._lock:
            if cls._instance is not None:
                # Signal the watcher thread to stop before clearing the instance.
                try:
                    cls._instance._stop_event.set()
                except Exception:
                    pass
                cls._instance = None
            cls._initialized = False
        logger.debug("ConfigManager singleton reset for testing.")

    def _check_config_updates(self):
        """Check if config files have been modified and reload if needed."""
        config_files = {
            "risk": "risk.yaml",
            "anpr": "anpr.yaml",
            "system": "system.yaml",
            "zones": "zones.yaml",
            "watchlist": "watchlist.yaml",
            "cameras": "cameras.yaml",
        }

        for key, filename in config_files.items():
            filepath = self.config_dir / filename
            if not filepath.exists():
                continue

            try:
                current_time = os.path.getmtime(filepath)
                last_time = self.file_times.get(key, 0)

                if current_time > last_time:
                    logger.info("Configuration file changed: %s", filename)
                    self._load_config_file(key, filepath)
                    self._notify_config_changed(key)
            except Exception as e:
                logger.exception("Unable to check configuration %s: %s", filename, e)

    def _notify_config_changed(self, config_key: str):
        """Notify all subscribers that a config changed."""
        if config_key in self.subscribers:
            for callback in self.subscribers[config_key]:
                try:
                    callback(config_key, self.configs[config_key])
                except Exception as e:
                    logger.exception("Configuration subscriber failed: %s", e)

        logger.info("Configuration updated: %s", config_key)

    def subscribe(self, config_key: str, callback):
        """Subscribe to config change notifications."""
        if config_key not in self.subscribers:
            self.subscribers[config_key] = []
        self.subscribers[config_key].append(callback)

    def get_config(self, key: str) -> Dict[str, Any]:
        """Get a config dictionary by key."""
        with self._file_lock:
            return self.configs.get(key, {}).copy() if self.configs.get(key) else {}

    def get_value(self, key: str, path: str, default: Any = None) -> Any:
        """Get a specific value from config using dot notation.

        Example:
            person_weight = config_manager.get_value("risk", "weights.person", 20)
        """
        config = self.get_config(key)
        parts = path.split(".")

        for part in parts:
            if isinstance(config, dict):
                config = config.get(part)
                if config is None:
                    return default
            else:
                return default

        return config if config is not None else default

    def reload_config(self, key: str) -> bool:
        """Force reload a specific config file."""
        filepath = self.config_dir / {
            "risk": "risk.yaml",
            "anpr": "anpr.yaml",
            "system": "system.yaml",
            "zones": "zones.yaml",
            "watchlist": "watchlist.yaml",
            "cameras": "cameras.yaml",
        }.get(key)

        if filepath and filepath.exists():
            self._load_config_file(key, filepath)
            return True
        return False


# Singleton instance
config_manager = ConfigManager()
