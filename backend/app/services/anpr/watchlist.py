import logging
from typing import Dict, Any
from backend.app.core.config import config_manager

logger = logging.getLogger("Watchlist")


class WatchlistService:
    def __init__(self, config_path: str = "configs/watchlist.yaml"):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Load watchlist config from config_manager."""
        watchlist_config = config_manager.get_config("watchlist")
        plates = watchlist_config.get("plates", []) if watchlist_config else []
        logger.info("Loaded %d plates into watchlist.", len(plates))

    def check_plate(self, plate: str) -> Dict[str, Any]:
        """
        Check if a plate is on the watchlist (fresh from config).
        Returns the match details or None.
        """
        # Always fetch fresh from config_manager to get latest watchlist
        watchlist_config = config_manager.get_config("watchlist")
        plates = watchlist_config.get("plates", []) if watchlist_config else []

        for entry in plates:
            if entry.get("plate") == plate:
                return entry
        return None


watchlist_service = WatchlistService()
