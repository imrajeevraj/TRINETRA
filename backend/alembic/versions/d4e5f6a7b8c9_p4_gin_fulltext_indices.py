"""P4 hardening: GIN full-text and trigram search indices

Revision ID: d4e5f6a7b8c9
Revises: c3f1a847e209
Create Date: 2026-09-02 00:45:00.000000

Changes:
  - P-03: PostgreSQL pg_trgm and GIN full-text indices for high-performance
    operator search over security_events (operator_notes, risk_reasons)
    and fast trigram substring matching on plate_events (plate_number).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'c3f1a847e209'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # P-03: Enable pg_trgm extension for trigram matching
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # GIN Full-Text Search index on security_events (operator_notes and risk_reasons)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_security_events_fts_gin "
        "ON security_events "
        "USING gin (to_tsvector('english', coalesce(operator_notes, '') || ' ' || coalesce(risk_reasons, '')))"
    )

    # GIN Trigram index on plate_events plate_number for fast substring search
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_plate_events_plate_number_trgm "
        "ON plate_events "
        "USING gin (plate_number gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_plate_events_plate_number_trgm")
    op.execute("DROP INDEX IF EXISTS ix_security_events_fts_gin")
