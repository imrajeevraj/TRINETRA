from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from backend.app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="VIEWER")  # ADMIN, OPERATOR, VIEWER
    created_at = Column(DateTime, default=datetime.utcnow)
