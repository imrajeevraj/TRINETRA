"""P4 hardening: performance indices and pgvector IVFFlat

Revision ID: c3f1a847e209
Revises: 820e9d52f121
Create Date: 2026-09-02 00:23:00.000000

Changes:
  - P-02: Individual and composite indices on security_events for all filtered
    query patterns used by /api/events/recent (camera_id, severity, status,
    timestamp, event_type, track_id).
  - P-05: IVFFlat approximate-nearest-neighbour index on face_embeddings.embedding
    for cosine similarity searches.  Uses lists=100 as a safe default for
    watchlists up to ~10 000 faces; reconfigure with REINDEX for larger datasets.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3f1a847e209'
down_revision: Union[str, Sequence[str], None] = '7b96ded74b21'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # P-02: security_events query indices
    op.create_index('ix_security_events_camera_id', 'security_events', ['camera_id'], unique=False)
    op.create_index('ix_security_events_severity', 'security_events', ['severity'], unique=False)
    op.create_index('ix_security_events_status', 'security_events', ['status'], unique=False)
    op.create_index('ix_security_events_timestamp', 'security_events', ['timestamp'], unique=False)
    op.create_index('ix_security_events_event_type', 'security_events', ['event_type'], unique=False)
    op.create_index('ix_security_events_track_id', 'security_events', ['track_id'], unique=False)

    # Composite index for the most common dashboard query:
    # data_origin + severity + timestamp DESC (recent high-severity live events)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_security_events_origin_severity_ts "
        "ON security_events (data_origin, severity, timestamp DESC)"
    )

    # P-05: IVFFlat approximate-nearest-neighbour index on face_embeddings.
    # lists=100 gives good recall for watchlists up to ~10 000 rows.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_face_embeddings_ivfflat "
        "ON face_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_face_embeddings_ivfflat")
    op.execute("DROP INDEX IF EXISTS ix_security_events_origin_severity_ts")
    op.drop_index('ix_security_events_track_id', table_name='security_events')
    op.drop_index('ix_security_events_event_type', table_name='security_events')
    op.drop_index('ix_security_events_timestamp', table_name='security_events')
    op.drop_index('ix_security_events_status', table_name='security_events')
    op.drop_index('ix_security_events_severity', table_name='security_events')
    op.drop_index('ix_security_events_camera_id', table_name='security_events')
