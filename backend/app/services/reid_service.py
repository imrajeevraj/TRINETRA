import logging
from sqlalchemy import select
from typing import Optional, Tuple
from datetime import datetime

from backend.app.core.database import SessionLocal
from backend.app.models.event import FaceEmbedding

logger = logging.getLogger("ReidService")


class ReidService:
    def __init__(self, match_threshold: float = 0.5):
        # Threshold for cosine distance (<=> operator in pgvector)
        # Lower means more similar.
        self.match_threshold = match_threshold

    def match_embedding(self, embedding_vector: list) -> Tuple[Optional[str], str]:
        """
        Takes a 512-d embedding vector and searches the database for a match.
        Returns a tuple of (person_name, watchlist_status).
        """
        db = SessionLocal()
        try:
            stmt_filtered = (
                select(FaceEmbedding)
                .filter(
                    FaceEmbedding.embedding.cosine_distance(embedding_vector)
                    < self.match_threshold
                )
                .order_by(FaceEmbedding.embedding.cosine_distance(embedding_vector))
                .limit(1)
            )

            matched = db.execute(stmt_filtered).scalars().first()
            if matched:
                # Found a match!
                # Update last seen
                matched.last_seen = datetime.utcnow()
                db.commit()
                return (
                    matched.person_name or f"Subject-{matched.id}",
                    matched.watchlist_status,
                )

            # If no match found, register as a new unknown face
            new_face = FaceEmbedding(
                embedding=embedding_vector, watchlist_status="UNKNOWN", person_name=None
            )
            db.add(new_face)
            db.commit()
            db.refresh(new_face)
            return f"Subject-{new_face.id}", "UNKNOWN"

        except Exception as e:
            logger.error(f"Error in ReID matching: {e}")
            db.rollback()
            return None, "ERROR"
        finally:
            db.close()


reid_service = ReidService()
