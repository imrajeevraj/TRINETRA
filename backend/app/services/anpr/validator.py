from collections import defaultdict
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("PlateValidator")


class PlateValidator:
    def __init__(self, required_hits: int = 3, consensus_ratio: float = 0.6):
        """
        Manages plate validations across multiple frames.
        Args:
            required_hits: Number of times a track must have a plate reading before we yield a result.
            consensus_ratio: Ratio of matching reads required to accept a plate (e.g., 0.6 = 60% of reads must match).
        """
        self.required_hits = required_hits
        self.consensus_ratio = consensus_ratio

        # Maps track_id -> list of (normalized_plate, confidence)
        self.track_history: Dict[str, list] = defaultdict(list)
        # Tracks that have already been validated and emitted an event, to avoid spam
        self.validated_tracks = set()

    def add_reading(
        self, track_id: str, plate_text: str, confidence: float
    ) -> Optional[Tuple[str, float]]:
        """
        Add a plate reading for a track.
        Returns (best_plate, avg_confidence) if consensus is reached, else None.
        """
        if track_id in self.validated_tracks:
            return None

        self.track_history[track_id].append((plate_text, confidence))

        history = self.track_history[track_id]
        if len(history) >= self.required_hits:
            return self._evaluate_consensus(track_id)

        return None

    def _evaluate_consensus(self, track_id: str) -> Optional[Tuple[str, float]]:
        history = self.track_history[track_id]

        # Count frequencies of each plate read
        counts = defaultdict(int)
        conf_sum = defaultdict(float)

        for plate, conf in history:
            counts[plate] += 1
            conf_sum[plate] += conf

        total_reads = len(history)

        # Find the most frequent read
        best_plate = None
        best_count = 0

        for plate, count in counts.items():
            if count > best_count:
                best_count = count
                best_plate = plate

        # Check if consensus ratio is met
        if best_count / total_reads >= self.consensus_ratio:
            avg_conf = conf_sum[best_plate] / best_count
            logger.info(
                f"Consensus reached for track {track_id}: {best_plate} (conf: {avg_conf:.2f}, hits: {best_count}/{total_reads})"
            )
            self.validated_tracks.add(track_id)
            # Free memory
            del self.track_history[track_id]
            return best_plate, avg_conf

        # If we have too many reads and no consensus, we might want to drop it or keep trying.
        # We'll just keep trying, but trim history to keep it bounded.
        if len(history) > self.required_hits * 3:
            # Keep the most recent ones
            self.track_history[track_id] = history[-self.required_hits :]

        return None

    def cleanup_track(self, track_id: str):
        """Remove a track from history when it leaves the scene."""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.validated_tracks:
            self.validated_tracks.remove(track_id)
