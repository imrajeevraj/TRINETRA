# IBVAP — Database Storage & Schema Migration Guidelines
# TRINETRA — Database Storage & Schema Migration Guidelines

## 1. Intentional Repository Exclusion
Runtime SQLite databases (`*.db`, `*.sqlite`, `*.sqlite3`), transaction logs (`*-wal`, `*-shm`), and PostgreSQL dump backups are **intentionally excluded from Git tracking**.

- **Security & Privacy:** Live runtime databases contain operational surveillance telemetry, audit logs, and potential operator credentials.
- **Code-Driven Architecture:** The state of the database must be reproducible purely from code migrations rather than binary database snapshots.

---

## 2. Schema Management & Migrations

All IBVAP database tables, relations, and vector indices are managed through **Alembic** migrations located in:
All TRINETRA database tables, relations, and vector indices are managed through **Alembic** migrations located in:
```
backend/alembic/versions/
├── e41c1a99996a_initial_schema.py        # Core tables: cameras, events, users, rules
├── 820e9d52f121_add_faceembedding.py       # Biometric face embedding vectors & pgvector index
├── 494432d1d9c5_add_data_origin.py         # Forensic chain-of-custody data origins
└── 7b96ded74b21_p2_forensics.py            # Phase 2 forensic audit tables
```

---

## 3. Initializing Local Database

### Option A: SQLite (Quick Local Development)
To initialize a fresh local SQLite database:
```bash
# From repository root
cd backend
alembic upgrade head
```
This creates a local `ibvap.db` in the repository root matching the latest production schema.

### Option B: PostgreSQL with pgvector (Production / Docker)
When running via Docker Compose:
```bash
docker compose up -d postgres redis
cd backend
alembic upgrade head
```
The database connection string is configured in `.env` via `DATABASE_URL`.
