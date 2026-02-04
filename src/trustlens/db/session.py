"""Database session helpers."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from trustlens.db.engine import build_engine


def get_session() -> Session:
    """Build and return a SQLAlchemy session bound to the project engine."""
    engine = build_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
