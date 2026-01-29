from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration loaded from environment variables and optional .env file.

    Key env var:
    - DATABASE_URL (SQLAlchemy URL)
      Default uses DuckDB file so the project runs immediately without Postgres.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "dev"
    database_url: str = "duckdb:///./data/trustlens.duckdb"


settings = Settings()
