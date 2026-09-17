"""Add audit_jobs table for R-05 persistent audit state

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-02 00:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'audit_jobs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('job_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('triggered_by', sa.String(), nullable=False),
        sa.Column('summary', sa.String(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('data_origin', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_jobs_id'), 'audit_jobs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_jobs_job_type'), 'audit_jobs', ['job_type'], unique=False)
    op.create_index(op.f('ix_audit_jobs_status'), 'audit_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_audit_jobs_data_origin'), 'audit_jobs', ['data_origin'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_jobs_data_origin'), table_name='audit_jobs')
    op.drop_index(op.f('ix_audit_jobs_status'), table_name='audit_jobs')
    op.drop_index(op.f('ix_audit_jobs_job_type'), table_name='audit_jobs')
    op.drop_index(op.f('ix_audit_jobs_id'), table_name='audit_jobs')
    op.drop_table('audit_jobs')
