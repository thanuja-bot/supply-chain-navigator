"""
Database setup using SQLAlchemy with SQLite (default) or PostgreSQL.

Engine creation is **lazy** — it only happens on first use, not at import time.
This prevents test environments from accidentally connecting to a production DB
when they supply their own sessions.

Configure by setting DATABASE_URL in your .env:
  SQLite (default): DATABASE_URL=sqlite:///./ai_pipeline.db
  PostgreSQL:       DATABASE_URL=postgresql://user:password@host:5432/dbname
"""

import os
from typing import Generator

from sqlalchemy import Column, DateTime, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    """Persisted run artifact row."""

    __tablename__ = "run_artifacts"

    run_id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(200), nullable=True, index=True)
    topic = Column(String(200), nullable=False)
    grade = Column(String(4), nullable=False)
    status = Column(String(20), nullable=False)
    artifact_json = Column(Text, nullable=False)  # Full RunArtifact as JSON
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Lazy engine — created on first call to get_engine()
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    """Return (or create) the singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL", "sqlite:///./ai_pipeline.db")
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, echo=False)
    return _engine


def get_session_factory():
    """Return (or create) the singleton session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


def init_db() -> None:
    """Create all tables. Called at application startup."""
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a SQLAlchemy session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
